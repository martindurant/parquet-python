
import io
import json
import numpy as np
import os
import pandas as pd
import shutil
import struct
import thriftpy

import numba

from thriftpy.protocol.compact import TCompactProtocolFactory
from .thrift_filetransport import TFileTransport
from .thrift_structures import parquet_thrift
from .compression import compress_data, decompress_data
from . import encoding
from .util import default_openw, default_mkdirs, sep_from_open

MARKER = b'PAR1'
NaT = np.timedelta64(None).tobytes()  # require numpy version >= 1.7

typemap = {  # primitive type, converted type, bit width
    'bool': (parquet_thrift.Type.BOOLEAN, None, 1),
    'int32': (parquet_thrift.Type.INT32, None, 32),
    'int64': (parquet_thrift.Type.INT64, None, 64),
    'int8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_8, 8),
    'int16': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_16, 16),
    'uint8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_8, 8),
    'uint16': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_16, 16),
    'uint32': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_32, 32),
    'float32': (parquet_thrift.Type.FLOAT, None, 32),
    'float64': (parquet_thrift.Type.DOUBLE, None, 64),
    'float16': (parquet_thrift.Type.FLOAT, None, 16),
}

revmap = {parquet_thrift.Type.INT32: np.int32,
          parquet_thrift.Type.INT64: np.int64,
          parquet_thrift.Type.FLOAT: np.float32}


def find_type(data, convert=False):
    """ Get appropriate typecodes for column dtype

    Data conversion may happen here, only at write time.

    The user is expected to transform their data into the appropriate dtype
    before saving to parquet, we will not make any assumptions for them.

    If the dtype is "object" the first ten items will be examined, and is str
    or bytes, will be stored as variable length byte strings; if dict or list,
    (nested data) will be stored with JSON encoding.

    To be stored as fixed-length byte strings, the dtype must be "bytesXX"
    (pandas notation) or "|SXX" (numpy notation)

    In the case of catagoricals, the data type refers to the labels; the data
    (codes) will be stored as int. The labels are usually variable length
    strings.

    BOOLs will be bitpacked using np.packbits. To instead keep the default numpy
    representation of one byte per value, change to int8 or uint8 type

    Known types that cannot be represented (must be first converted another
    type or to raw binary): float128, complex

    Parameters
    ----------
    A pandas series.

    Returns
    -------
    - a thrift schema element
    - a thrift typecode to be passed to the column chunk writer

    """
    out = None
    dtype = data.dtype
    if dtype.name in typemap:
        type, converted_type, width = typemap[dtype.name]
        if type in revmap and convert:
            out = data.values.astype(revmap[type])
        elif type == parquet_thrift.Type.BOOLEAN and convert:
            padded = np.lib.pad(data.values, (0, 8 - (len(data) % 8)),
                                'constant', constant_values=(0, 0))
            out = np.packbits(padded.reshape(-1, 8)[:, ::-1].ravel())
        elif convert:
            out = data.values
    elif "S" in str(dtype) or "U" in str(dtype):
        # TODO: check effect of unicode
        type, converted_type, width = (parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY,
                                       None, dtype.itemsize)
        if convert:
            out = data.values
    elif dtype == "O":
        head = data[:10] if isinstance(data, pd.Index) else data.valid()[:10]
        if all(isinstance(i, str) for i in head):
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.UTF8, None)
            if convert:
                out = data.str.encode('utf8').values
        elif all(isinstance(i, bytes) for i in head):
            type, converted_type, width = parquet_thrift.Type.BYTE_ARRAY, None, None
            if convert:
                out = data.values
        elif all(isinstance(i, (list, dict)) for i in head):
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.JSON, None)
            if convert:
                out = data.map(json.dumps).str.encode('utf8').values
        else:
            raise ValueError("Data type conversion unknown: %s" % dtype)
    elif str(dtype).startswith("datetime64"):
        type, converted_type, width = (parquet_thrift.Type.INT64,
                                       parquet_thrift.ConvertedType.TIMESTAMP_MICROS, None)
        if convert:
            out = data.values.astype('datetime64[us]')
    elif str(dtype).startswith("timedelta64"):
        type, converted_type, width = (parquet_thrift.Type.INT64,
                                       parquet_thrift.ConvertedType.TIME_MICROS, None)
        if convert:
            out = data.values.astype('timedelta64[us]')
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    # TODO: pandas has no explicit support for Decimal
    se = parquet_thrift.SchemaElement(name=data.name, type_length=width,
                                      converted_type=converted_type, type=type,
                                      repetition_type=parquet_thrift.FieldRepetitionType.REQUIRED)
    return se, type, out


def thrift_print(structure, offset=0):
    """
    Handy recursive text ouput for thrift structures
    """
    if not isinstance(structure, thriftpy.thrift.TPayload):
        return str(structure)
    s = str(structure.__class__) + '\n'
    for key in dir(structure):
        if key.startswith('_') or key in ['thrift_spec', 'read', 'write',
                                          'default_spec']:
            continue
        s = s + ' ' * offset + key + ': ' + thrift_print(getattr(structure, key)
                                                         , offset+2) + '\n'
    return s
thriftpy.thrift.TPayload.__str__ = thrift_print
thriftpy.thrift.TPayload.__repr__ = thrift_print


def write_thrift(fobj, thrift):
    """Write binary compact representation of thiftpy structured object

    Parameters
    ----------
    fobj: open file-like object (binary mode)
    thrift: thriftpy object to write

    Returns
    -------
    Number of bytes written
    """
    t0 = fobj.tell()
    tout = TFileTransport(fobj)
    pout = TCompactProtocolFactory().get_protocol(tout)
    thrift.write(pout)
    return fobj.tell() - t0


def encode_plain(data, se):
    """PLAIN encoding; returns byte representation"""
    se, type, out = find_type(data, True)
    if data.dtype == "O":
        return b''.join([struct.pack('<l', len(x)) + x for x in out])
    else:
        return out.tobytes()


@numba.njit(nogil=True)
def encode_unsigned_varint(x, o):
    while x > 127:
        o.write_byte((x & 0x7F) | 0x80)
        x >>= 7
    o.write_byte(x)


@numba.jit(nogil=True)
def zigzag(n):
    " 32-bit only "
    return (n << 1) ^ (n >> 31)


@numba.njit(nogil=True)
def encode_bitpacked_inv(values, width, o):
    bit = 16 - width
    right_byte_mask = 0b11111111
    left_byte_mask = right_byte_mask << 8
    bits = 0
    for v in values:
        bits |= v << bit
        while bit <= 8:
            o.write_byte((bits & left_byte_mask) >> 8)
            bit += 8
            bits = (bits & right_byte_mask) << 8
        bit -= width
    if bit:
        o.write_byte((bits & left_byte_mask) >> 8)


@numba.njit(nogil=True)
def encode_bitpacked(values, width, o):
    """
    Write values packed into width-bits each (which can be >8)

    values is a NumbaIO array (int32)
    o is a NumbaIO output array (uint8), size=(len(values)*width)/8, rounded up.
    """
    bit_packed_count = (len(values) + 7) // 8
    encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header

    bit = 0
    right_byte_mask = 0b11111111
    bits = 0
    for v in values:
        bits |= v << bit
        bit += width
        while bit >= 8:
            o.write_byte(bits & right_byte_mask)
            bit -= 8
            bits >>= 8
    if bit:
        o.write_byte(bits)


def write_length(l, o):
    """ Put a 32-bit length into four bytes in o

    Equivalent to struct.pack('<i', l), but suitable for numba-jit
    """
    right_byte_mask = 0b11111111
    for _ in range(4):
        o.write_byte(l & right_byte_mask)
        l >>= 8


def encode_rle_bp(data, width, o, withlength=False):
    """Write data into o using RLE/bitpacked hybrid

    data : values to encode (int32)
    width : bits-per-value, set by max(data)
    o : output encoding.Numpy8
    withlength : bool
        If definitions/repetitions, length of data must be pre-written
    """
    if withlength:
        start = o.loc
        o.loc = o.loc + 4
    if True:
        # I don't know how one would choose between RLE and bitpack
        encode_bitpacked(data, width, o)
    if withlength:
        end = o.loc
        o.loc = start
        write_length(wnd - start, o)
        o.loc = end


def encode_dict(data, se):
    """ The data part of dictionary encoding is always int32s, with RLE/bitpack
    """
    width = encoding.width_from_max_int(data.max())
    ldata = ((len(data) + 7) // 8) * width + 11
    i = data.values.astype(np.int32)
    out = encoding.Numpy8(np.empty(ldata, dtype=np.uint8))
    out.write_byte(width)
    encode_rle_bp(i, width, out)
    return out.so_far().tostring()

encode = {
    'PLAIN': encode_plain,
    'RLE': encode_rle_bp,
    'PLAIN_DICTIONARY': encode_dict,
    # 'DELTA_BINARY_PACKED': encode_delta
}


def make_definitions(data):
    """For data that can contain NULLs, produce definition levels binary
    data: either bitpacked bools, or (if number of nulls == 0), single RLE
    block."""
    valid = ~data.isnull()
    temp = encoding.Numpy8(np.empty(10, dtype=np.uint8))

    if valid.all():
        # no nulls at all
        l = len(valid)
        encode_unsigned_varint(l << 1, temp)
        temp.write_byte(1)
        block = struct.pack('<i', temp.loc) + temp.so_far().tostring()
    else:
        # bitpack bools
        out = encode_plain(valid, None)

        encode_unsigned_varint(len(out) << 1 | 1, temp)
        head = temp.so_far().tostring()

        block = struct.pack('<i', len(head + out)) + head + out
    return block, data.valid()


def write_column(f, data, selement, encoding='PLAIN', compression=None):
    """
    Write a single column of data to an open Parquet file

    Parameters
    ----------
    f: open binary file
    data: pandas Series or numpy (1d) array
    selement: thrift SchemaElement
        produced by ``find_type``
    encoding: one of ``parquet_thift.Encoding``
        if the dtype is categorical, this is ignored and dictionary encoding
        automatically used
    compression: str or None
        if not None, must be one of the keys in ``compression.compress``

    Returns
    -------
    chunk: ColumnChunk structure

    """
    has_nulls = selement.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    tot_rows = len(data)

    # no NULL handling (but NaNs, NaTs are allowed)
    if has_nulls:
        print('has nulls!')
        definition_data, data = make_definitions(data)
    else:
        definition_data = b""

    # No nested field handling (encode those as J/BSON)
    repetition_data = b""

    rows = len(data)
    cats = False
    name = data.name
    diff = 0

    if str(data.dtype) == 'category':
        dph = parquet_thrift.DictionaryPageHeader(
                num_values=len(data.cat.categories),
                encoding=parquet_thrift.Encoding.PLAIN)
        bdata = encode['PLAIN'](pd.Series(data.cat.categories), selement)
        l0 = len(bdata)
        if compression:
            bdata = compress_data(bdata, compression)
            l1 = len(bdata)
        else:
            l1 = l0
        diff += l0 - l1
        ph = parquet_thrift.PageHeader(
                type=parquet_thrift.PageType.DICTIONARY_PAGE,
                uncompressed_page_size=l0, compressed_page_size=l1,
                dictionary_page_header=dph, crc=None)

        dict_start = f.tell()
        write_thrift(f, ph)
        f.write(bdata)
        data = data.cat.codes.astype(np.int32)
        cats = True
        encoding = "PLAIN_DICTIONARY"

    start = f.tell()
    bdata = definition_data + repetition_data + encode[encoding](data, selement)
    try:
        max, min = data.max(), data.min()
        if encoding == "DELTA_BINARY_PACKED":
            encode2 = "PLAIN"
        else:
            encode2 = encoding
        max = encode[encode2](pd.Series([max], dtype=data.dtype), selement)
        min = encode[encode2](pd.Series([min], dtype=data.dtype), selement)
    except TypeError:
        max, min = None, None

    dph = parquet_thrift.DataPageHeader(
            num_values=tot_rows,
            encoding=getattr(parquet_thrift.Encoding, encoding),
            definition_level_encoding=parquet_thrift.Encoding.RLE,
            repetition_level_encoding=parquet_thrift.Encoding.BIT_PACKED)
    l0 = len(bdata)

    if compression:
        bdata = compress_data(bdata, compression)
        l1 = len(bdata)
    else:
        l1 = l0
    diff += l0 - l1

    ph = parquet_thrift.PageHeader(type=parquet_thrift.PageType.DATA_PAGE,
                                   uncompressed_page_size=l0,
                                   compressed_page_size=l1,
                                   data_page_header=dph, crc=None)

    write_thrift(f, ph)
    f.write(bdata)

    compressed_size = f.tell() - start
    uncompressed_size = compressed_size + diff

    offset = f.tell()
    s = parquet_thrift.Statistics(max=max, min=min, null_count=0)

    p = [parquet_thrift.PageEncodingStats(
            page_type=parquet_thrift.PageType.DATA_PAGE,
            encoding=parquet_thrift.Encoding.PLAIN, count=1)]

    cmd = parquet_thrift.ColumnMetaData(
            type=selement.type, path_in_schema=[name],
            encodings=[parquet_thrift.Encoding.RLE,
                       parquet_thrift.Encoding.BIT_PACKED,
                       parquet_thrift.Encoding.PLAIN],
            codec=getattr(parquet_thrift.CompressionCodec, compression.upper()) if compression else 0,
            num_values=tot_rows,
            statistics=s,
            data_page_offset=start,
            encoding_stats=p,
            key_value_metadata=[],
            total_uncompressed_size=uncompressed_size,
            total_compressed_size=compressed_size)
    if cats:
        p.append(parquet_thrift.PageEncodingStats(
                page_type=parquet_thrift.PageType.DICTIONARY_PAGE,
                encoding=parquet_thrift.Encoding.PLAIN, count=1))
        cmd.dictionary_page_offset = dict_start
    chunk = parquet_thrift.ColumnChunk(file_offset=offset,
                                       meta_data=cmd)
    write_thrift(f, chunk)
    return chunk


def make_row_group(f, data, schema, file_path=None, compression=None,
                   encoding='PLAIN'):
    """ Make a single row group of a Parquet file """
    rows = len(data)
    rg = parquet_thrift.RowGroup(num_rows=rows, total_byte_size=0, columns=[])

    for column in schema:
        if column.type is not None:
            if isinstance(compression, dict):
                comp = compression.get(column.name, None)
            else:
                comp = compression
            chunk = write_column(f, data[column.name], column,
                                 compression=comp, encoding=encoding)
            rg.columns.append(chunk)
    rg.total_byte_size = sum([c.meta_data.total_uncompressed_size for c in
                              rg.columns])

    return rg


def make_part_file(f, data, schema, compression=None, encoding='PLAIN'):
    with f as f:
        f.write(MARKER)
        rg = make_row_group(f, data, schema, compression=compression,
                            encoding=encoding)
        fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                          schema=schema,
                                          version=1,
                                          created_by='parquet-python',
                                          row_groups=[rg])
        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)
    return rg


def make_metadata(data, has_nulls=[]):
    root = parquet_thrift.SchemaElement(name='schema',
                                        num_children=0)

    fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                      schema=[root],
                                      version=1,
                                      created_by='fastparquet-python',
                                      row_groups=[])

    for column in data.columns:
        if str(data[column].dtype) == 'category':
            se, type, _ = find_type(data[column].cat.categories)
            se.name = column
        else:
            se, type, _ = find_type(data[column])
        if column in has_nulls and str(data[column].dtype) in ['category', 'object']:
            se.repetition_type = parquet_thrift.FieldRepetitionType.OPTIONAL
        fmd.schema.append(se)
        root.num_children += 1
    return fmd


def write(filename, data, partitions=[0], encoding="PLAIN",
          compression=None, file_scheme='simple', open_with=default_openw,
          mkdirs=default_mkdirs, has_nulls=[], write_index=None):
    """ Write Pandas DataFrame to filename as Parquet Format

    Parameters
    ----------
    filename: string
        File contains everything (if file_scheme='same'), else contains the
        metadata only
    data: pandas dataframe or groupby
        If a dataframe, will create one file per partition; if a groupby,
        will create partitioned directory tree and one or more files in
        each location - the file_scheme will then be hive. Note that for
        groupby objects, null values in the original grouping columns will
        be silently dropped.
    partitions: list of row index values to start new row groups
    encoding: single value from parquet_thrift.Encoding, if applied to all
        columns, or dict of name:parquet_thrift.Encoding for a different
        encoding per column.
    compression: str, dict
        compression to apply to each column, e.g. GZIP or SNAPPY
    file_scheme: 'simple'|'hive'
        If simple: all goes in a single file
        If hive: each row group is in a separate file, and filename contains
        only the metadata
    open_with: function
        When called with a path/URL, returns an open file-like object
    mkdirs: function
        When called with a path/URL, creates any necessary dictionaries to
        make that location writable, e.g., ``os.makedirs``. This is not
        necessary if using the simple file scheme
    has_nulls: list of strings
        The named columns can have nulls. Only applies to Object and Category
        columns, as pandas ints can't have NULLs, and NaN/NaT is equivalent
        to NULL in float and time-like columns.
    write_index: boolean
        Whether or not to write the index to a separate column.  By default we
        write the index *if* it is not 0, 1, ..., n.

    Examples
    --------
    >>> fastparquet.write('myfile.parquet', df)  # doctest: +SKIP
    """
    sep = sep_from_open(open_with)
    if file_scheme != 'simple' or isinstance(data, pd.core.groupby.DataFrameGroupBy):
        mkdirs(filename)
        fn = sep.join([filename, '_metadata'])
    else:
        fn = filename
    with open_with(fn) as f:
        f.write(MARKER)

        if isinstance(data, pd.core.groupby.DataFrameGroupBy):
            fmd = False
            names = list(data.dtypes.index.names)
            for key in data.indices:
                df = data.get_group(key)
                fmd = fmd or make_metadata(df, has_nulls=has_nulls)
                path = sep.join(["%s=%s" % (name, val) for name, val in zip(names, key)])
                mkdirs(filename+sep+path)
                for i, start in enumerate(partitions):
                    end = partitions[i+1] if i < (len(partitions) - 1) else None
                    part = 'part.%i.parquet' % i
                    partname = sep.join([filename, path, part])
                    with open_with(partname) as f2:
                        rg = make_part_file(f2, df[start:end], fmd.schema,
                                            compression=compression,
                                            encoding=encoding)
                    for chunk in rg.columns:
                        chunk.file_path = sep.join([path, part])

                    fmd.row_groups.append(rg)
            fmd.num_rows = sum(rg.num_rows for rg in fmd.row_groups)
        else:
            if write_index or write_index is None and not (
                    isinstance(data.index, pd.RangeIndex) and
                    data.index._start == 0 and
                    data.index._stop == len(data.index) and
                    data.index._step == 1 and data.index.name is None):
                data = data.reset_index()
            fmd = make_metadata(data, has_nulls=has_nulls)
            for i, start in enumerate(partitions):
                end = partitions[i+1] if i < (len(partitions) - 1) else None
                if file_scheme == 'simple':
                    rg = make_row_group(f, data[start:end], fmd.schema,
                                        compression=compression, encoding=encoding)
                else:
                    part = 'part.%i.parquet' % i
                    partname = sep.join([filename, part])
                    with open_with(partname) as f2:
                        rg = make_part_file(f2, data[start:end], fmd.schema,
                                            compression=compression,
                                            encoding=encoding)
                    for chunk in rg.columns:
                        chunk.file_path = part

                fmd.row_groups.append(rg)

        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)

    if file_scheme != 'simple':
        write_common_metadata(sep.join([filename, '_common_metadata']), fmd,
                              open_with)


def write_common_metadata(fn, fmd, open_with=default_openw):
    """For hive-style parquet, write schema in special shared file"""
    if isinstance(fn, str):
        f = open_with(fn)
    else:
        f = fn
    f.write(MARKER)
    fmd.row_groups = []
    foot_size = write_thrift(f, fmd)
    f.write(struct.pack(b"<i", foot_size))
    f.write(MARKER)
    f.close()

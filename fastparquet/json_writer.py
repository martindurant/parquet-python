from __future__ import print_function

import json
import re
import struct
import warnings

import numba
import numpy as np
import pandas as pd
from six import integer_types, text_type

from fastparquet.encoding import width_from_max_int, rle_bit_packed_hybrid
from fastparquet.parquet_thrift.parquet.ttypes import FieldRepetitionType, Type
from fastparquet.util import join_path
from jx_base import STRING, OBJECT, NESTED, python_type_to_json_type
from mo_dots import concat_field, split_field, startswith_field, coalesce
from mo_future import sort_using_key
from mo_json.typed_encoder import NESTED_TYPE
from pyLibrary.env.typed_inserter import json_type_to_inserter_type
from .thrift_structures import write_thrift

try:
    from pandas.api.types import is_categorical_dtype
except ImportError:
    # Pandas <= 0.18.1
    from pandas.core.common import is_categorical_dtype
from .thrift_structures import parquet_thrift
from .compression import compress_data
from .converted_types import tobson
from . import encoding, api
from .util import (default_open, default_mkdirs,
                   index_like, PY2, STR_TYPE,
                   check_column_names, metadata_from_many, created_by,
                   get_column_metadata)
from .speedups import array_encode_utf8, pack_byte_array

MARKER = b'PAR1'
NaT = np.timedelta64(None).tobytes()  # require numpy version >= 1.7
nat = np.datetime64('NaT').view('int64')

typemap = {  # primitive type, converted type, bit width
    'bool': (parquet_thrift.Type.BOOLEAN, None, 1),
    'int32': (parquet_thrift.Type.INT32, None, 32),
    'int64': (parquet_thrift.Type.INT64, None, 64),
    'int8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_8, 8),
    'int16': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_16, 16),
    'uint8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_8, 8),
    'uint16': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_16, 16),
    'uint32': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_32, 32),
    'uint64': (parquet_thrift.Type.INT64, parquet_thrift.ConvertedType.UINT_64, 64),
    'float32': (parquet_thrift.Type.FLOAT, None, 32),
    'float64': (parquet_thrift.Type.DOUBLE, None, 64),
    'float16': (parquet_thrift.Type.FLOAT, None, 16),
}

revmap = {parquet_thrift.Type.INT32: np.int32,
          parquet_thrift.Type.INT64: np.int64,
          parquet_thrift.Type.FLOAT: np.float32,
          parquet_thrift.Type.DOUBLE: np.float64}


def find_type(data, fixed_text=None, object_encoding=None, times='int64'):
    """ Get appropriate typecodes for column dtype

    Data conversion do not happen here, see convert().

    The user is expected to transform their data into the appropriate dtype
    before saving to parquet, we will not make any assumptions for them.

    Known types that cannot be represented (must be first converted another
    type or to raw binary): float128, complex

    Parameters
    ----------
    data: pd.Series
    fixed_text: int or None
        For str and bytes, the fixed-string length to use. If None, object
        column will remain variable length.
    object_encoding: None or infer|bytes|utf8|json|bson|bool|int|int32|float
        How to encode object type into bytes. If None, bytes is assumed;
        if 'infer', type is guessed from 10 first non-null values.
    times: 'int64'|'int96'
        Normal integers or 12-byte encoding for timestamps.

    Returns
    -------
    - a thrift schema element
    - a thrift typecode to be passed to the column chunk writer
    - converted data (None if convert is False)

    """
    dtype = data.dtype
    if dtype.name in typemap:
        type, converted_type, width = typemap[dtype.name]
    elif "S" in str(dtype)[:2] or "U" in str(dtype)[:2]:
        type, converted_type, width = (parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY,
                                       None, dtype.itemsize)
    elif dtype == "O":
        if object_encoding == 'infer':
            object_encoding = infer_object_encoding(data)

        if object_encoding == 'utf8':
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.UTF8,
                                           None)
        elif object_encoding in ['bytes', None]:
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY, None,
                                           None)
        elif object_encoding == 'json':
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.JSON,
                                           None)
        elif object_encoding == 'bson':
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.BSON,
                                           None)
        elif object_encoding == 'bool':
            type, converted_type, width = (parquet_thrift.Type.BOOLEAN, None,
                                           1)
        elif object_encoding == 'int':
            type, converted_type, width = (parquet_thrift.Type.INT64, None,
                                           64)
        elif object_encoding == 'int32':
            type, converted_type, width = (parquet_thrift.Type.INT32, None,
                                           32)
        elif object_encoding == 'float':
            type, converted_type, width = (parquet_thrift.Type.DOUBLE, None,
                                           64)
        else:
            raise ValueError('Object encoding (%s) not one of '
                             'infer|utf8|bytes|json|bson|bool|int|int32|float' %
                             object_encoding)
        if fixed_text:
            width = fixed_text
            type = parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY
    elif dtype.kind == "M":
        if times == 'int64':
            type, converted_type, width = (
                parquet_thrift.Type.INT64,
                parquet_thrift.ConvertedType.TIMESTAMP_MICROS, None)
        elif times == 'int96':
            type, converted_type, width = (parquet_thrift.Type.INT96, None,
                                           None)
        else:
            raise ValueError(
                    "Parameter times must be [int64|int96], not %s" % times)
        if hasattr(dtype, 'tz') and str(dtype.tz) != 'UTC':
            warnings.warn('Coercing datetimes to UTC')
    elif dtype.kind == "m":
        type, converted_type, width = (parquet_thrift.Type.INT64,
                                       parquet_thrift.ConvertedType.TIME_MICROS, None)
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    se = parquet_thrift.SchemaElement(
            name=data.name, type_length=width,
            converted_type=converted_type, type=type,
            repetition_type=parquet_thrift.FieldRepetitionType.REQUIRED)
    return se, type


def convert(data, se):
    """Convert data according to the schema encoding"""
    dtype = data.values.dtype
    type = se.type
    converted_type = se.converted_type
    if dtype.name in typemap:
        if type in revmap:
            out = data.values.astype(revmap[type], copy=False)
        elif type == parquet_thrift.Type.BOOLEAN:
            padded = np.lib.pad(data.values, (0, 8 - (len(data.values) % 8)),
                                'constant', constant_values=(0, 0))
            out = np.packbits(padded.reshape(-1, 8)[:, ::-1].ravel())
        elif dtype.name in typemap:
            out = data.values
    elif "S" in str(dtype)[:2] or "U" in str(dtype)[:2]:
        out = data.values
    elif dtype == "O":
        try:
            if converted_type == parquet_thrift.ConvertedType.UTF8:
                out = array_encode_utf8(data)
            elif converted_type is None:
                if type in revmap:
                    out = data.values.astype(revmap[type], copy=False)
                elif type == parquet_thrift.Type.BOOLEAN:
                    padded = np.lib.pad(data.values, (0, 8 - (len(data) % 8)),
                                        'constant', constant_values=(0, 0))
                    out = np.packbits(padded.reshape(-1, 8)[:, ::-1].ravel())
                else:
                    out = data.values
            elif converted_type == parquet_thrift.ConvertedType.JSON:
                out = np.array([json.dumps(x).encode('utf8') for x in data],
                               dtype="O")
            elif converted_type == parquet_thrift.ConvertedType.BSON:
                out = data.map(tobson).values
            if type == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
                out = out.astype('S%i' % se.type_length)
        except Exception as e:
            ct = parquet_thrift.ConvertedType._VALUES_TO_NAMES[
                converted_type] if converted_type is not None else None
            raise ValueError('Error converting column "%s" to bytes using '
                             'encoding %s. Original error: '
                             '%s' % (data.name, ct, e))
    elif converted_type == parquet_thrift.ConvertedType.TIMESTAMP_MICROS:
        out = np.empty(len(data), 'int64')
        time_shift(data.values.view('int64'), out)
    elif converted_type == parquet_thrift.ConvertedType.TIME_MICROS:
        out = np.empty(len(data), 'int64')
        time_shift(data.values.view('int64'), out)
    elif type == parquet_thrift.Type.INT96 and dtype.kind == 'M':
        ns_per_day = (24 * 3600 * 1000000000)
        day = data.values.view('int64') // ns_per_day + 2440588
        ns = (data.values.view('int64') % ns_per_day)# - ns_per_day // 2
        out = np.empty(len(data), dtype=[('ns', 'i8'), ('day', 'i4')])
        out['ns'] = ns
        out['day'] = day
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    return out


def infer_object_encoding(data):
    head = data[:10] if isinstance(data, pd.Index) else data.valid()[:10]
    if all(isinstance(i, STR_TYPE) for i in head) and not PY2:
        return "utf8"
    elif PY2 and all(isinstance(i, unicode) for i in head):
        return "utf8"
    elif all(isinstance(i, STR_TYPE) for i in head) and PY2:
        return "bytes"
    elif all(isinstance(i, bytes) for i in head):
        return 'bytes'
    elif all(isinstance(i, (list, dict)) for i in head):
        return 'json'
    elif all(isinstance(i, bool) for i in head):
        return 'bool'
    elif all(isinstance(i, integer_types) for i in head):
        return 'int'
    elif all(isinstance(i, float) or isinstance(i, np.floating)
             for i in head):
        # You need np.floating here for pandas NaNs in object
        # columns with python floats.
        return 'float'
    else:
        raise ValueError("Can't infer object conversion type: %s" % head)



def time_shift(indata, outdata, factor=1000):  # pragma: no cover
    for i in range(len(indata)):
        if indata[i] == nat:
            outdata[i] = nat
        else:
            outdata[i] = indata[i] // factor


def encode_plain(data, se):
    """PLAIN encoding; returns byte representation"""
    out = convert(data, se)
    if se.type == parquet_thrift.Type.BYTE_ARRAY:
        return pack_byte_array(list(out))
    else:
        return out.tobytes()



def encode_unsigned_varint(x, o):  # pragma: no cover
    while x > 127:
        o.write_byte((x & 0x7F) | 0x80)
        x >>= 7
    o.write_byte(x)


@numba.jit(nogil=True)
def zigzag(n):  # pragma: no cover
    " 32-bit only "
    return (n << 1) ^ (n >> 31)



def encode_bitpacked_inv(values, width, o):  # pragma: no cover
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



def encode_bitpacked(values, width, o):  # pragma: no cover
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


def encode_rle(data, se, fixed_text=None):
    if data.dtype.kind not in ['i', 'u']:
        raise ValueError('RLE/bitpack encoding only works for integers')
    if se.type_length in [8, 16]:
        o = encoding.Numpy8(np.empty(10, dtype=np.uint8))
        bit_packed_count = (len(data) + 7) // 8
        encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header
        return o.so_far().tostring() + data.values.tostring()
    else:
        m = data.max()
        width = width_from_max_int(m)
        l = (len(data) * width + 7) // 8 + 10
        o = encoding.Numpy8(np.empty(l, dtype='uint8'))
        encode_rle_bp(data, width, o)
        return o.so_far().tostring()


def encode_dict(data, se):
    """ The data part of dictionary encoding is always int8, with RLE/bitpack
    """
    width = data.values.dtype.itemsize * 8
    o = encoding.Numpy8(np.empty(10, dtype=np.uint8))
    o.write_byte(width)
    bit_packed_count = (len(data) + 7) // 8
    encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header
    return o.so_far().tostring() + data.values.tostring()

encode = {
    'PLAIN': encode_plain,
    'RLE': encode_rle,
    'PLAIN_DICTIONARY': encode_dict,
    # 'DELTA_BINARY_PACKED': encode_delta
}


def make_definitions(data, no_nulls):
    """For data that can contain NULLs, produce definition levels binary
    data: either bitpacked bools, or (if number of nulls == 0), single RLE
    block."""
    temp = encoding.Numpy8(np.empty(10, dtype=np.uint8))

    if no_nulls:
        # no nulls at all
        l = len(data)
        encode_unsigned_varint(l << 1, temp)  # lsb marks the data as bit-packed-run as per https://github.com/apache/parquet-format/blob/master/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3
        temp.write_byte(1)
        block = struct.pack('<i', temp.loc) + temp.so_far().tostring()
        out = data
    else:
        se = parquet_thrift.SchemaElement(type=parquet_thrift.Type.BOOLEAN)
        out = encode_plain(data.notnull(), se)

        encode_unsigned_varint(len(out) << 1 | 1, temp)
        head = temp.so_far().tostring()

        block = struct.pack('<i', len(head + out)) + head + out
        out = data.valid()  # better, data[data.notnull()], from above ?
    return block, out


def write_column(f, data, selement, compression=None):
    """
    Write a single column of data to an open Parquet file

    Parameters
    ----------
    f: open binary file
    data: pandas Series or numpy (1d) array
    selement: thrift SchemaElement
        produced by ``find_type``
    compression: str or None
        if not None, must be one of the keys in ``compression.compress``

    Returns
    -------
    chunk: ColumnChunk structure

    """
    has_nulls = selement.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    tot_rows = len(data)
    encoding = "PLAIN"

    if is_categorical_dtype(data.values.dtype):
        num_nulls = (data.cat.codes == -1).sum()
    elif data.values.dtype.kind in ['i', 'b']:
        num_nulls = 0
    else:
        num_nulls = len(data) - data.values.count()

    bit_width = width_from_max_int(data.max_definition_level)
    definition_data = rle_bit_packed_hybrid(data.def_levels, bit_width)
    repetition_data = rle_bit_packed_hybrid(data.rep_levels, bit_width)

    if data.values.dtype.kind == "O" and not is_categorical_dtype(data.values.dtype):
        try:
            if selement.type == parquet_thrift.Type.INT64:
                data = data.values.astype(int)
            elif selement.type == parquet_thrift.Type.BOOLEAN:
                data = data.values.astype(bool)
        except ValueError as e:
            t = parquet_thrift.Type._VALUES_TO_NAMES[selement.type]
            raise ValueError('Error converting column "%s" to primitive '
                             'type %s. Original error: '
                             '%s' % (data.values.name, t, e))

    cats = False
    name = data.values.name
    diff = 0
    max, min = None, None

    if is_categorical_dtype(data.values.dtype):
        dph = parquet_thrift.DictionaryPageHeader(
                num_values=len(data.values.cat.categories),
                encoding=parquet_thrift.Encoding.PLAIN)
        bdata = encode['PLAIN'](pd.Series(data.values.cat.categories), selement)
        bdata += 8 * b'\x00'
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
        try:
            if num_nulls == 0:
                max, min = data.values.values.max(), data.values.values.min()
                if selement.type == parquet_thrift.Type.BYTE_ARRAY:
                    if selement.converted_type is not None:
                        max = encode['PLAIN'](pd.Series([max]), selement)[4:]
                        min = encode['PLAIN'](pd.Series([min]), selement)[4:]
                else:
                    max = encode['PLAIN'](pd.Series([max]), selement)
                    min = encode['PLAIN'](pd.Series([min]), selement)
        except TypeError:
            pass
        ncats = len(data.values.cat.categories)
        data = data.values.cat.codes
        cats = True
        encoding = "PLAIN_DICTIONARY"
    elif str(data.values.dtype) in ['int8', 'int16', 'uint8', 'uint16']:
        encoding = "RLE"

    start = f.tell()
    bdata = definition_data + repetition_data + encode[encoding](
            data.values, selement)
    bdata += 8 * b'\x00'
    try:
        if encoding != 'PLAIN_DICTIONARY' and num_nulls == 0:
            max, min = data.values.values.max(), data.values.values.min()
            if selement.type == parquet_thrift.Type.BYTE_ARRAY:
                if selement.converted_type is not None:
                    max = encode['PLAIN'](pd.Series([max]), selement)[4:]
                    min = encode['PLAIN'](pd.Series([min]), selement)[4:]
            else:
                max = encode['PLAIN'](pd.Series([max]), selement)
                min = encode['PLAIN'](pd.Series([min]), selement)
    except TypeError:
        pass

    dph = parquet_thrift.DataPageHeader(
        num_values=tot_rows,
        encoding=getattr(parquet_thrift.Encoding, encoding),
        definition_level_encoding=parquet_thrift.Encoding.RLE,
        repetition_level_encoding=parquet_thrift.Encoding.RLE
    )
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
    s = parquet_thrift.Statistics(max=max, min=min, null_count=num_nulls)

    p = [parquet_thrift.PageEncodingStats(
            page_type=parquet_thrift.PageType.DATA_PAGE,
            encoding=parquet_thrift.Encoding.PLAIN, count=1)]

    cmd = parquet_thrift.ColumnMetaData(
            type=selement.type, path_in_schema=[name],
            encodings=[parquet_thrift.Encoding.RLE,
                       parquet_thrift.Encoding.BIT_PACKED,
                       parquet_thrift.Encoding.PLAIN],
            codec=(getattr(parquet_thrift.CompressionCodec, compression.upper())
                   if compression else 0),
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
        cmd.key_value_metadata.append(
            parquet_thrift.KeyValue(key='num_categories', value=str(ncats)))
        cmd.key_value_metadata.append(
            parquet_thrift.KeyValue(key='numpy_dtype', value=str(data.values.dtype)))
    chunk = parquet_thrift.ColumnChunk(file_offset=offset,
                                       meta_data=cmd)
    write_thrift(f, chunk)
    return chunk


def make_row_group(f, data, schema, compression=None):
    """ Make a single row group of a Parquet file """
    rows = len(data)
    if rows == 0:
        return
    if any(not isinstance(c, (bytes, STR_TYPE)) for c in data.values):
        raise ValueError('Column names must be str or bytes:',
                         {c: type(c) for c in data.values.columns
                          if not isinstance(c, (bytes, STR_TYPE))})
    rg = parquet_thrift.RowGroup(num_rows=rows, total_byte_size=0, columns=[])

    for column in schema:
        if column.type is not None:
            if isinstance(compression, dict):
                comp = compression.get(column.name, None)
            else:
                comp = compression
            chunk = write_column(f, data.get_column(column.name), column,
                                 compression=comp)
            rg.columns.append(chunk)
    rg.total_byte_size = sum([c.meta_data.total_uncompressed_size for c in
                              rg.columns])
    return rg


def make_part_file(f, data, schema, compression=None, fmd=None):
    if len(data) == 0:
        return
    with f as f:
        f.write(MARKER)
        rg = make_row_group(f, data, schema, compression=compression)
        if fmd is None:
            fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                              schema=schema,
                                              version=1,
                                              created_by=created_by,
                                              row_groups=[rg])
            foot_size = write_thrift(f, fmd)
            f.write(struct.pack(b"<i", foot_size))
        else:
            prev = fmd.row_groups
            fmd.row_groups = [rg]
            foot_size = write_thrift(f, fmd)
            f.write(struct.pack(b"<i", foot_size))
            fmd.row_groups = prev
        f.write(MARKER)
    return rg


def make_metadata(data, has_nulls=True, ignore_columns=[], fixed_text=None,
                  object_encoding=None, times='int64', index_cols=[]):
    if not data.columns.is_unique:
        raise ValueError('Cannot create parquet dataset with duplicate'
                         ' column names (%s)' % data.columns)
    pandas_metadata = {'index_columns': index_cols,
                       'columns': [], 'pandas_version': pd.__version__}
    root = parquet_thrift.SchemaElement(name='schema',
                                        num_children=0)

    meta = parquet_thrift.KeyValue()
    meta.key = 'pandas'
    output = parquet_thrift.FileMetaData(num_rows=len(data),
                                      schema=[root],
                                      version=1,
                                      created_by=created_by,
                                      row_groups=[],
                                      key_value_metadata=[meta])

    object_encoding = object_encoding or {}
    for column in data.columns:
        if column in ignore_columns:
            continue
        pandas_metadata['columns'].append(
            get_column_metadata(data[column], column))
        oencoding = (object_encoding if isinstance(object_encoding, STR_TYPE)
                     else object_encoding.get(column, None))
        fixed = None if fixed_text is None else fixed_text.get(column, None)
        if is_categorical_dtype(data[column].dtype):
            se, type = find_type(data[column].cat.categories,
                                 fixed_text=fixed, object_encoding=oencoding)
            se.name = column
        else:
            se, type = find_type(data[column], fixed_text=fixed,
                                 object_encoding=oencoding, times=times)
        col_has_nulls = has_nulls
        if has_nulls is None:
            se.repetition_type = data[column].dtype == "O"
        elif has_nulls is not True and has_nulls is not False:
            col_has_nulls = column in has_nulls
        if col_has_nulls:
            se.repetition_type = parquet_thrift.FieldRepetitionType.OPTIONAL
        output.schema.append(se)
        root.num_children += 1
    meta.value = json.dumps(pandas_metadata, sort_keys=True)
    return output


def write_simple(fn, data, fmd, row_group_offsets, compression,
                 open_with, has_nulls, append=False):
    """
    Write to one single file (for file_scheme='simple')
    """
    if append:
        pf = api.ParquetFile(fn, open_with=open_with)
        if pf.file_scheme not in ['simple', 'empty']:
            raise ValueError('File scheme requested is simple, but '
                             'existing file scheme is not')
        fmd = pf.fmd
        mode = 'rb+'
    else:
        mode = 'wb'
    with open_with(fn, mode) as f:
        if append:
            f.seek(-8, 2)
            head_size = struct.unpack('<i', f.read(4))[0]
            f.seek(-(head_size+8), 2)
        else:
            f.write(MARKER)
        for i, start in enumerate(row_group_offsets):
            end = (row_group_offsets[i+1] if i < (len(row_group_offsets) - 1)
                   else None)
            rg = make_row_group(f, data[start:end], fmd.schema,
                                compression=compression)
            if rg is not None:
                fmd.row_groups.append(rg)

        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)


def write(filename, data, row_group_offsets=50000000,
          compression=None, file_scheme='simple', open_with=default_open,
          mkdirs=default_mkdirs, has_nulls=True, write_index=None,
          partition_on=[], fixed_text=None, append=False,
          object_encoding='infer', times='int64'):
    """ Write Pandas DataFrame to filename as Parquet Format

    Parameters
    ----------
    filename: string
        Parquet collection to write to, either a single file (if file_scheme
        is simple) or a directory containing the metadata and data-files.
    data: pandas dataframe
        The table to write
    row_group_offsets: int or list of ints
        If int, row-groups will be approximately this many rows, rounded down
        to make row groups about the same size; if a list, the explicit index
        values to start new row groups.
    compression: str, dict
        compression to apply to each column, e.g. GZIP or SNAPPY or
        {col1: "SNAPPY", col2: None} to specify per column.
    file_scheme: 'simple'|'hive'
        If simple: all goes in a single file
        If hive: each row group is in a separate file, and a separate file
        (called "_metadata") contains the metadata.
    open_with: function
        When called with a f(path, mode), returns an open file-like object
    mkdirs: function
        When called with a path/URL, creates any necessary dictionaries to
        make that location writable, e.g., ``os.makedirs``. This is not
        necessary if using the simple file scheme
    has_nulls: bool, 'infer' or list of strings
        Whether columns can have nulls. If a list of strings, those given
        columns will be marked as "optional" in the metadata, and include
        null definition blocks on disk. Some data types (floats and times)
        can instead use the sentinel values NaN and NaT, which are not the same
        as NULL in parquet, but functionally act the same in many cases,
        particularly if converting back to pandas later. A value of 'infer'
        will assume nulls for object columns and not otherwise.
    write_index: boolean
        Whether or not to write the index to a separate column.  By default we
        write the index *if* it is not 0, 1, ..., n.
    partition_on: list of column names
        Passed to groupby in order to split data within each row-group,
        producing a structured directory tree. Note: as with pandas, null
        values will be dropped. Ignored if file_scheme is simple.
    fixed_text: {column: int length} or None
        For bytes or str columns, values will be converted
        to fixed-length strings of the given length for the given columns
        before writing, potentially providing a large speed
        boost. The length applies to the binary representation *after*
        conversion for utf8, json or bson.
    append: bool (False)
        If False, construct data-set from scratch; if True, add new row-group(s)
        to existing data-set. In the latter case, the data-set must exist,
        and the schema must match the input data.
    object_encoding: str or {col: type}
        For object columns, this gives the data type, so that the values can
        be encoded to bytes. Possible values are bytes|utf8|json|bson|bool|int|int32,
        where bytes is assumed if not specified (i.e., no conversion). The
        special value 'infer' will cause the type to be guessed from the first
        ten non-null values.
    times: 'int64' (default), or 'int96':
        In "int64" mode, datetimes are written as 8-byte integers, us
        resolution; in "int96" mode, they are written as 12-byte blocks, with
        the first 8 bytes as ns within the day, the next 4 bytes the julian day.
        'int96' mode is included only for compatibility.

    Examples
    --------
    >>> fastparquet.write('myfile.parquet', df)  # doctest: +SKIP
    """
    schema = SchemaTree()
    data, new_schama = rows_to_columns(data, schema)

    if str(has_nulls) == 'infer':
        has_nulls = None
    if isinstance(row_group_offsets, int):
        l = len(data)
        nparts = max((l - 1) // row_group_offsets + 1, 1)
        chunksize = max(min((l - 1) // nparts + 1, l), 1)
        row_group_offsets = list(range(0, l, chunksize))
    if write_index or write_index is None and index_like(data.values.index):
        cols = set(data.values)
        data = data.reset_index()
        index_cols = [c for c in data if c not in cols]
    else:
        index_cols = []
    check_column_names(data.values.columns, partition_on, fixed_text, object_encoding,
                       has_nulls)
    ignore = partition_on if file_scheme != 'simple' else []
    fmd = make_metadata(data.values, has_nulls=has_nulls, ignore_columns=ignore,
                        fixed_text=fixed_text, object_encoding=object_encoding,
                        times=times, index_cols=index_cols)

    if file_scheme == 'simple':
        write_simple(filename, data, fmd, row_group_offsets,
                     compression, open_with, has_nulls, append)
    elif file_scheme in ['hive', 'drill']:
        if append:
            pf = api.ParquetFile(filename, open_with=open_with)
            if pf.file_scheme not in ['hive', 'empty', 'flat']:
                raise ValueError('Requested file scheme is %s, but '
                                 'existing file scheme is not.' % file_scheme)
            fmd = pf.fmd
            i_offset = find_max_part(fmd.row_groups)
            if tuple(partition_on) != tuple(pf.cats):
                raise ValueError('When appending, partitioning columns must'
                                 ' match existing data')
        else:
            i_offset = 0
        fn = join_path(filename, '_metadata')
        mkdirs(filename)
        for i, start in enumerate(row_group_offsets):
            end = (row_group_offsets[i+1] if i < (len(row_group_offsets) - 1)
                   else None)
            part = 'part.%i.parquet' % (i + i_offset)
            if partition_on:
                rgs = partition_on_columns(
                    data[start:end], partition_on, filename, part, fmd,
                    compression, open_with, mkdirs,
                    with_field=file_scheme == 'hive'
                )
                fmd.row_groups.extend(rgs)
            else:
                partname = join_path(filename, part)
                with open_with(partname, 'wb') as f2:
                    rg = make_part_file(f2, data[start:end], fmd.schema,
                                        compression=compression, fmd=fmd)
                for chunk in rg.columns:
                    chunk.file_path = part

                fmd.row_groups.append(rg)

        write_common_metadata(fn, fmd, open_with, no_row_groups=False)
        write_common_metadata(join_path(filename, '_common_metadata'), fmd,
                              open_with)
    else:
        raise ValueError('File scheme should be simple|hive, not', file_scheme)


def find_max_part(row_groups):
    """
    Find the highest integer matching "**part.*.parquet" in referenced paths.
    """
    paths = [c.file_path or "" for rg in row_groups for c in rg.columns]
    s = re.compile('.*part.(?P<i>[\d]+).parquet$')
    matches = [s.match(path) for path in paths]
    nums = [int(match.groupdict()['i']) for match in matches if match]
    if nums:
        return max(nums) + 1
    else:
        return 0


def partition_on_columns(data, columns, root_path, partname, fmd,
                         compression, open_with, mkdirs, with_field=True):
    """
    Split each row-group by the given columns

    Each combination of column values (determined by pandas groupby) will
    be written in structured directories.
    """
    gb = data.groupby(columns)
    remaining = list(data)
    for column in columns:
        remaining.remove(column)
    if not remaining:
        raise ValueError("Cannot include all columns in partition_on")
    rgs = []
    for key in sorted(gb.indices):
        df = gb.get_group(key)[remaining]
        if not isinstance(key, tuple):
            key = (key,)
        if with_field:
            path = join_path(*(
                "%s=%s" % (name, val)
                for name, val in zip(columns, key)
            ))
        else:
            path = join_path(*("%s" % val for val in key))
        relname = join_path(path, partname)
        mkdirs(join_path(root_path, path))
        fullname = join_path(root_path, path, partname)
        with open_with(fullname, 'wb') as f2:
            rg = make_part_file(f2, df, fmd.schema,
                                compression=compression, fmd=fmd)
        if rg is not None:
            for chunk in rg.columns:
                chunk.file_path = relname
            rgs.append(rg)
    return rgs


def write_common_metadata(fn, fmd, open_with=default_open,
                          no_row_groups=True):
    """
    For hive-style parquet, write schema in special shared file

    Parameters
    ----------
    fn: str
        Filename to write to
    fmd: thrift FileMetaData
        Information to write
    open_with: func
        To use to create writable file as f(path, mode)
    no_row_groups: bool (True)
        Strip out row groups from metadata before writing - used for "common
        metadata" files, containing only the schema.
    """
    consolidate_categories(fmd)
    with open_with(fn, 'wb') as f:
        f.write(MARKER)
        if no_row_groups:
            rgs = fmd.row_groups
            fmd.row_groups = []
            foot_size = write_thrift(f, fmd)
            fmd.row_groups = rgs
        else:
            foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)


def consolidate_categories(fmd):
    key_value = [k for k in fmd.key_value_metadata
                 if k.key == 'pandas'][0]
    meta = json.loads(key_value.value)
    cats = [c for c in meta['columns']
            if 'num_categories' in (c['metadata'] or [])]
    for cat in cats:
        for rg in fmd.row_groups:
            for col in rg.columns:
                if ".".join(col.meta_data.path_in_schema) == cat['name']:
                    ncats = [k.value for k in col.meta_data.key_value_metadata
                             if k.key == 'num_categories']
                    if ncats and int(ncats[0]) > cat['metadata'][
                            'num_categories']:
                        cat['metadata']['num_categories'] = int(ncats[0])
    key_value.value = json.dumps(meta, sort_keys=True)


def merge(file_list, verify_schema=True, open_with=default_open,
          root=False):
    """
    Create a logical data-set out of multiple parquet files.

    The files referenced in file_list must either be in the same directory,
    or at the same level within a structured directory, where the directories
    give partitioning information. The schemas of the files should also be
    consistent.

    Parameters
    ----------
    file_list: list of paths or ParquetFile instances
    verify_schema: bool (True)
        If True, will first check that all the schemas in the input files are
        identical.
    open_with: func
        Used for opening a file for writing as f(path, mode). If input list
        is ParquetFile instances, will be inferred from the first one of these.
    root: str
        If passing a list of files, the top directory of the data-set may
        be ambiguous for partitioning where the upmost field has only one
        value. Use this to specify the data'set root directory, if required.

    Returns
    -------
    ParquetFile instance corresponding to the merged data.
    """
    basepath, fmd = metadata_from_many(file_list, verify_schema, open_with,
                                       root=root)

    out_file = join_path(basepath, '_metadata')
    write_common_metadata(out_file, fmd, open_with, no_row_groups=False)
    out = api.ParquetFile(out_file, open_with=open_with)

    out_file = join_path(basepath, '_common_metadata')
    write_common_metadata(out_file, fmd, open_with)
    return out


def rows_to_columns(data, schema):
    """
    REPETITION LEVELS DO NOT REQUIRE MORE THAN A LIST OF COLUMNS TO FILL
    :param data: array of objects
    :param schema: Known schema
    :return: values, repetition levels, new schema
    """

    new_schema = []

    all_leaves = schema.leaves
    values = {full_name: [] for full_name in all_leaves}
    rep_levels = {full_name: [] for full_name in all_leaves}
    def_levels = {full_name: [] for full_name in all_leaves}

    def _none_to_column(schema, path, rep_level, counters):
        if schema:
            for name, sub_schema in schema.items():
                new_path = concat_field(path, name)
                _none_to_column(sub_schema, new_path, rep_level, counters)
        else:
            values[path].append(None)
            rep_levels[path].append(rep_level)
            def_levels[path].append(len(counters)-1)

    def _value_to_column(value, schema, path, counters):
        ptype = type(value)
        dtype, jtype, itype = python_type_to_all_types[ptype]
        if jtype is NESTED:
            new_path = concat_field(path, NESTED_TYPE)
            sub_schema = schema.more.get(NESTED_TYPE)
            if not sub_schema:
                sub_schema = schema.more[NESTED_TYPE] = SchemaTree()

            if not value:
                _none_to_column(sub_schema, new_path, get_rep_level(counters), counters)
            else:
                for k, new_value in enumerate(value):
                    new_counters = counters + (k,)
                    _value_to_column(new_value, sub_schema, new_path, new_counters)
        elif jtype is OBJECT:
            if not value:
                _none_to_column(schema, path, get_rep_level(counters), counters)
            else:
                for name, sub_schema in schema.more.items():
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    _value_to_column(new_value, sub_schema, new_path, counters)

                for name in set(value.keys())-set(schema.more.keys()):
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    sub_schema = schema.more[name] = SchemaTree()
                    _value_to_column(new_value, sub_schema, new_path, counters)
        else:
            typed_name = concat_field(path, itype)
            if jtype is STRING:
                value = value.encode('utf8')
            element, is_new = merge_schema_element(schema.values.get(itype), typed_name, value, ptype, dtype, jtype, itype)
            if is_new:
                schema.values[itype] = element
                new_schema.append(element)
                values[typed_name] = [None] * counters[0]
                rep_levels[typed_name] = [0] * counters[0]
                def_levels[typed_name] = [0] * counters[0]
            values[typed_name].append(value)
            rep_levels[typed_name].append(get_rep_level(counters))
            def_levels[typed_name].append(len(counters) - 1)

    for rownum, new_value in enumerate(data):
        _value_to_column(new_value, schema, '.', (rownum,))

    return Table(values, rep_levels, def_levels, len(data), schema), new_schema


def get_rep_level(counters):
    for rep_level, c in reversed(list(enumerate(counters))):
        if c > 0:
            return rep_level
    return 0  # SHOULD BE -1 FOR MISSING RECORD, BUT WE WILL ASSUME THE RECORD EXISTS


def merge_schema_element(element, name, value, ptype, dtype, jtype, ittype):
    if not element:
        output = parquet_thrift.SchemaElement(
            type=dtype,
            type_length=get_length(value, dtype),
            repetition_type=get_repetition_type(jtype),
            name=name
        )
        return output, True
    else:
        element.type_length = max(element.type_length, get_length(value, dtype))

        return element, False


def get_length(value, dtype):
    if dtype is Type.BYTE_ARRAY:
        return len(value)
    elif dtype is None:
        return 0
    else:
        return 8


def get_repetition_type(jtype):
    return FieldRepetitionType.REPEATED if jtype is NESTED else FieldRepetitionType.OPTIONAL


class Table(object):
    """
    REPRESENT A DATA CUBE
    """

    def __init__(self, values, rep_levels, def_levels, num_rows, schema):
        """
        :param values: dict from full name to list of values
        :param rep_levels:  dict from full name to list of values
        :param def_levels: dict from full name to list of values
        :param num_rows: number of rows in the dataset
        :param schema: The complete SchemaTree
        """
        self.values = pd.DataFrame.from_dict(values)
        self.rep_levels = pd.DataFrame.from_dict(rep_levels)
        self.def_levels = pd.DataFrame.from_dict(def_levels)
        self.num_rows = num_rows
        self.schema = schema
        self.max_definition_level = schema.max_definition_level()

    def __getattr__(self, item):
        return getattr(self.values, item)

    def get_column(self, item):
        sub_schema=self.schema
        for n in split_field(item):
            if n in sub_schema.more:
                sub_schema = sub_schema.more.get(n)
            else:
                sub_schema = sub_schema.values.get(n)

        return Column(
            item,
            self.values[item],
            self.rep_levels[item],
            self.def_levels[item],
            self.num_rows,
            sub_schema,
            self.max_definition_level
        )

    def __getitem__(self, item):
        if isinstance(item, text_type):
            sub_schema=self.schema
            for n in split_field(item):
                if n in sub_schema.more:
                    sub_schema = sub_schema.more.get(n)
                else:
                    sub_schema =sub_schema.values.get(n)

            return Table(
                {k: v for k, v in self.values.items() if startswith_field(k, item)},
                {k: v for k, v in self.rep_levels.items() if startswith_field(k, item)},
                {k: v for k, v in self.def_levels.items() if startswith_field(k, item)},
                self.num_rows,
                sub_schema
            )
        elif isinstance(item, slice):
            start = coalesce(item.start, 0)
            stop = coalesce(item.stop, self.num_rows)

            if start == 0 and stop == self.num_rows:
                return self

            first = 0
            last = 0
            counter = 0
            for i, r in enumerate(self.rep_levels):
                if counter == start:
                    first = i
                elif counter == stop:
                    last = i
                    break
                if r == 0:
                    counter += 1

            return Table(
                {k: v[first:last] for k, v in self.values.items()},
                {k: v[first:last] for k, v in self.rep_levels.items()},
                {k: v[first:last] for k, v in self.def_levels.items()},
                stop-start,
                self.schema
            )

    def __len__(self):
        return self.num_rows

class Column(object):
    """
    REPRESENT A DATA FRAME
    """

    def __init__(self, name, values, rep_levels, def_levels, num_rows, schema, max_definition_level):
        """
        :param values: MAP FROM NAME TO LIST OF PARQUET VALUES
        :param schema:
        """
        self.name = name
        self.values = values
        self.rep_levels = rep_levels
        self.def_levels = def_levels
        self.num_rows = num_rows
        self.schema = schema
        self.max_definition_level = max_definition_level

    def __len__(self):
        return self.num_rows


class SchemaTree(object):

    def __init__(self):
        self.more = {}  # MAP FROM NAME TO MORE SchemaTree
        self.values = {}  # MAP FROM JSON TYPE TO SchemaElement

    @property
    def leaves(self):
        return [
            json_type_to_inserter_type[jtype]
            for jtype in self.values.keys()
        ]+[
            concat_field(name, leaf)
            for name, child_schema in self.more.items()
            for leaf in child_schema.leaves
        ]

    def get_parquet_metadata(self, path='.'):
        """
        OUTPUT PARQUET METADATA COLUMNS
        :param path: FOR INTERNAL USE
        :return: LIST OF SchemaElement
        """
        children = []
        for name, child_schema in sort_using_key(self.more.items(), lambda p: p[0]):
            children.extend(child_schema.get_parquet_metadata(concat_field(path, name)))
        children.extend(v for k, v in sort_using_key(self.values.items(), lambda p: p[0]))
        return [parquet_thrift.SchemaElement(
            name=path,
            num_children=sum(coalesce(c.num_children, 0) + 1 for c in children)
        )] + children

    def max_definition_level(self):
        if not self.more:
            return 1
        else:
            max_child = [m.max_definition_level() for m in self.more.values()]
            return max(max_child) + 1


python_type_to_parquet_type = {
    bool: Type.BOOLEAN,
    text_type: Type.BYTE_ARRAY,
    int: Type.INT64,
    float: Type.DOUBLE,
    dict: None,
    list: None
}

if PY2:
    python_type_to_parquet_type[long] = Type.INT64

# MAP FROM PYTHON TYPE TO (parquet_type, json_type, inserter_type)
python_type_to_all_types = {
    ptype: (dtype, python_type_to_json_type[ptype], json_type_to_inserter_type.get(python_type_to_json_type[ptype]))
    for ptype, dtype in python_type_to_parquet_type.items()
}



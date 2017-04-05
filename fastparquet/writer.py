from __future__ import print_function

import io
import json
import numpy as np
import os
import pandas as pd
import re
import shutil
import struct
import sys
import thriftpy
import warnings
import time

import numba

from thriftpy.protocol.compact import TCompactProtocolFactory
from thriftpy.protocol.exc import TProtocolException
from .thrift_filetransport import TFileTransport
from .thrift_structures import parquet_thrift
from .compression import compress_data, decompress_data
from .converted_types import tobson
from . import encoding, api
from .util import (default_open, default_mkdirs, sep_from_open,
                   ParquetException, thrift_copy, index_like, PY2, STR_TYPE,
                   check_column_names, metadata_from_many, created_by)
from .speedups import array_encode_utf8, pack_byte_array
from . import quantize

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
          parquet_thrift.Type.FLOAT: np.float32}


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
    object_encoding: None or bytes|utf8\json|bson
        How to encode object type into bytes. If None, bytes is assumed;
        if 'infer'
    times: 'int64'|'int96'
        Normal integers or 12-byte encoding for timestamps.

    Returns
    -------
    - a thrift schema element
    - a thrift typecode to be passed to the column chunk writer
    - converted data (None if convert is False)

    """
    out = None
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
                                           parquet_thrift.ConvertedType.UTF8, None)
        elif object_encoding in ['bytes', None]:
            type, converted_type, width = parquet_thrift.Type.BYTE_ARRAY, None, None
        elif object_encoding == 'json':
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.JSON, None)
        elif object_encoding == 'bson':
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.BSON, None)
        else:
            raise ValueError('Object encoding (%s) not one of '
                             'infer|utf8|bytes|json|bson' % object_encoding)
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
    dtype = data.dtype
    type = se.type
    converted_type = se.converted_type
    if dtype.name in typemap:
        if type in revmap:
            out = data.values.astype(revmap[type], copy=False)
        elif type == parquet_thrift.Type.BOOLEAN:
            padded = np.lib.pad(data.values, (0, 8 - (len(data) % 8)),
                                'constant', constant_values=(0, 0))
            out = np.packbits(padded.reshape(-1, 8)[:, ::-1].ravel())
        elif dtype.name in typemap:
            out = data.values
    elif "S" in str(dtype)[:2] or "U" in str(dtype)[:2]:
        out = data.values
    elif dtype == "O":
        if converted_type == parquet_thrift.ConvertedType.UTF8:
            out = array_encode_utf8(data)
        elif converted_type is None:
            out = data.values
        elif converted_type == parquet_thrift.ConvertedType.JSON:
            out = np.array([json.dumps(x).encode('utf8') for x in data],
                           dtype="O")
        elif converted_type == parquet_thrift.ConvertedType.BSON:
            out = data.map(tobson).values
        if type == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
            out = out.astype('S%i' % se.type_length)
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
    if PY2 and all(isinstance(i, unicode) for i in head):
        return "utf8"
    if all(isinstance(i, STR_TYPE) for i in head) and PY2:
        return "bytes"
    if all(isinstance(i, bytes) for i in head):
        return 'bytes'
    if all(isinstance(i, (list, dict)) for i in head):
        return 'json'
    else:
        raise ValueError("Can't infer object conversion type: %s" % head)


@numba.njit(nogil=True)
def time_shift(indata, outdata, factor=1000):  # pragma: no cover
    for i in range(len(indata)):
        if indata[i] == nat:
            outdata[i] = nat
        else:
            outdata[i] = indata[i] // factor


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
    try:
        thrift.write(pout)
        fail = False
    except TProtocolException as e:
        typ, val, tb = sys.exc_info()
        frames = []
        while tb is not None:
            frames.append(tb)
            tb = tb.tb_next
        frame = [tb for tb in frames if 'write_struct' in str(tb.tb_frame.f_code)]
        variables = frame[0].tb_frame.f_locals
        obj = variables['obj']
        name = variables['fname']
        fail = True
    if fail:
        raise ParquetException('Thrift parameter validation failure %s'
                               ' when writing: %s-> Field: %s' % (
            val.args[0], obj, name
        ))
    return fobj.tell() - t0


def encode_plain(data, se):
    """PLAIN encoding; returns byte representation"""
    out = convert(data, se)
    if se.type == parquet_thrift.Type.BYTE_ARRAY:
        return pack_byte_array(list(out))
    else:
        return out.tobytes()


@numba.njit(nogil=True)
def encode_unsigned_varint(x, o):  # pragma: no cover
    while x > 127:
        o.write_byte((x & 0x7F) | 0x80)
        x >>= 7
    o.write_byte(x)


@numba.jit(nogil=True)
def zigzag(n):  # pragma: no cover
    " 32-bit only "
    return (n << 1) ^ (n >> 31)


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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
        width = 0
        while m:
            m >>= 1
            width += 1
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


# Cache the median absolute deviation of columns here per call to `write`.
COLUMN_MADS = {}

def encode_lossy_rice(data, se, quantization_level, quantization_random_state):
    global COLUMN_MADS

    if data.name not in COLUMN_MADS:
        # Really good idea from J. Michelson to use every n-th value
        # to speed up MAD computation.
        # Using it here to keep just 10k values for MAD computation.
        if data.shape[0] > 10000:
            modval = data.shape[0] // 10000
            vals = data.values[::modval]
        else:
            vals = data.values

        # Use median abolsute deviation to estimate scale of distribution,
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='All-NaN', category=RuntimeWarning, append=True)
            scale = (np.nanmedian(np.abs(vals - np.nanmedian(vals))) *
                     1.4826 /
                     quantization_level)
            scale = scale if scale > 0.0 else 1.0

        # Cache it to save time for later row groups.
        COLUMN_MADS[data.name] = scale
    else:
        scale = COLUMN_MADS[data.name]
    zero_point = np.nanmin(data.values) - scale * (-2147483647 + 10)
    seed = quantization_random_state.randint(1, 100000000)

    if 'float64' in str(data.values.dtype):
        return quantize.quantize_and_compress_float64(
            np.ascontiguousarray(data.values), scale, zero_point, seed)
    else:
        return quantize.quantize_and_compress_float32(
            np.ascontiguousarray(data.values), scale, zero_point, seed)

encode = {
    'PLAIN': encode_plain,
    'RLE': encode_rle,
    'PLAIN_DICTIONARY': encode_dict,
    # 'DELTA_BINARY_PACKED': encode_delta
    'LOSSY_RICE': encode_lossy_rice,
}


def make_definitions(data, no_nulls):
    """For data that can contain NULLs, produce definition levels binary
    data: either bitpacked bools, or (if number of nulls == 0), single RLE
    block."""
    temp = encoding.Numpy8(np.empty(10, dtype=np.uint8))

    if no_nulls:
        # no nulls at all
        l = len(data)
        encode_unsigned_varint(l << 1, temp)
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


def write_column(f, data, selement, compression=None,
                 quantization_level=None, quantization_random_state=None):
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
    quantization_level: int or None
        If not None, floating point values are quantized in units of the median
        absolute deviation divided by `quantization_level`. These integer values are then RICE
        encoded.
    quantization_random_state: `np.random.RandomState` or None
        The RNG used to dither the floating point values.

    Returns
    -------
    chunk: ColumnChunk structure

    """
    has_nulls = selement.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    tot_rows = len(data)
    encoding = "PLAIN"

    if has_nulls:
        if str(data.dtype) == 'category':
            num_nulls = (data.cat.codes == -1).sum()
        elif data.dtype.kind == 'i':
            num_nulls = 0
        else:
            num_nulls = len(data) - data.count()
        definition_data, data = make_definitions(data, num_nulls == 0)
    else:
        definition_data = b""
        num_nulls = 0

    # No nested field handling (encode those as J/BSON)
    repetition_data = b""

    cats = False
    name = data.name
    diff = 0
    max, min = None, None

    if str(data.dtype) == 'category':
        dph = parquet_thrift.DictionaryPageHeader(
                num_values=len(data.cat.categories),
                encoding=parquet_thrift.Encoding.PLAIN)
        bdata = encode['PLAIN'](pd.Series(data.cat.categories), selement)
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
                max, min = data.values.max(), data.values.min()
                max = encode['PLAIN'](pd.Series([max]), selement)
                min = encode['PLAIN'](pd.Series([min]), selement)
        except TypeError:
            pass
        data = data.cat.codes
        cats = True
        encoding = "PLAIN_DICTIONARY"
    elif str(data.dtype) in ['int8', 'int16', 'uint8', 'uint16']:
        encoding = "RLE"
    elif str(data.dtype) in ['float32', 'float64'] and quantization_level is not None:
        encoding = "LOSSY_RICE"

    start = f.tell()
    encoding_args = [data, selement]
    if encoding == 'LOSSY_RICE':
        encoding_args.extend([quantization_level, quantization_random_state])
    bdata = definition_data + repetition_data + encode[encoding](*encoding_args)
    bdata += 8 * b'\x00'
    try:
        if encoding != 'PLAIN_DICTIONARY' and num_nulls == 0:
            max, min = data.values.max(), data.values.min()
            max = encode['PLAIN'](pd.Series([max], dtype=data.dtype), selement)
            min = encode['PLAIN'](pd.Series([min], dtype=data.dtype), selement)
    except TypeError:
        pass

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
    chunk = parquet_thrift.ColumnChunk(file_offset=offset,
                                       meta_data=cmd)
    write_thrift(f, chunk)
    return chunk


def make_row_group(f, data, schema, compression=None,
                   quantization_level=None, quantization_random_state=None):
    """ Make a single row group of a Parquet file """
    rows = len(data)
    if rows == 0:
        return
    if any(not isinstance(c, (bytes, STR_TYPE)) for c in data):
        raise ValueError('Column names must be str or bytes:',
                         {c: type(c) for c in data.columns
                          if not isinstance(c, (bytes, STR_TYPE))})
    rg = parquet_thrift.RowGroup(num_rows=rows, total_byte_size=0, columns=[])

    for column in schema:
        if column.type is not None:
            if isinstance(compression, dict):
                comp = compression.get(column.name, None)
            else:
                comp = compression
            chunk = write_column(
                f, data[column.name], column,
                compression=comp,
                quantization_level=quantization_level,
                quantization_random_state=quantization_random_state)
            rg.columns.append(chunk)
    rg.total_byte_size = sum([c.meta_data.total_uncompressed_size for c in
                              rg.columns])
    return rg


def make_part_file(f, data, schema, compression=None):
    if len(data) == 0:
        return
    with f as f:
        f.write(MARKER)
        rg = make_row_group(f, data, schema, compression=compression)
        fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                          schema=schema,
                                          version=1,
                                          created_by=created_by,
                                          row_groups=[rg])
        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)
    return rg


def make_metadata(data, has_nulls=True, ignore_columns=[], fixed_text=None,
                  object_encoding=None, times='int64'):
    if not data.columns.is_unique:
        raise ValueError('Cannot create parquet dataset with duplicate'
                         ' column names (%s)' % data.columns)
    root = parquet_thrift.SchemaElement(name='schema',
                                        num_children=0)

    cats = parquet_thrift.KeyValue()
    cats.key = 'fastparquet.cats'
    catstruct = {}
    fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                      schema=[root],
                                      version=1,
                                      created_by=created_by,
                                      row_groups=[],
                                      key_value_metadata=[cats])

    object_encoding = object_encoding or {}
    for column in data.columns:
        if column in ignore_columns:
            continue
        oencoding = (object_encoding if isinstance(object_encoding, STR_TYPE) else
                     object_encoding.get(column, None))
        fixed = None if fixed_text is None else fixed_text.get(column, None)
        if str(data[column].dtype) == 'category':
            se, type = find_type(data[column].cat.categories,
                                 fixed_text=fixed, object_encoding=oencoding)
            se.name = column
            catstruct[column] = len(data[column].cat.categories)
        else:
            se, type = find_type(data[column], fixed_text=fixed,
                                 object_encoding=oencoding, times=times)
        col_has_nulls = has_nulls
        if has_nulls is None:
            se.repetition_type = type in [parquet_thrift.Type.BYTE_ARRAY,
                                          parquet_thrift.Type.INT96]
        elif has_nulls is not True and has_nulls is not False:
            col_has_nulls = column in has_nulls
        if col_has_nulls and data[column].dtype.kind != 'i':
            se.repetition_type = parquet_thrift.FieldRepetitionType.OPTIONAL
        fmd.schema.append(se)
        root.num_children += 1
    cats.value = json.dumps(catstruct)
    return fmd


def write_simple(fn, data, fmd, row_group_offsets, compression,
                 open_with, has_nulls, append=False,
                 quantization_level=None, quantization_random_state=None):
    """
    Write to one single file (for file_scheme='simple')
    """
    if append:
        pf = api.ParquetFile(fn, open_with=open_with)
        if pf.file_scheme != 'simple':
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
                                compression=compression,
                                quantization_level=quantization_level,
                                quantization_random_state=quantization_random_state)
            if rg is not None:
                fmd.row_groups.append(rg)

        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)


def write(filename, data, row_group_offsets=50000000,
          compression=None, file_scheme='simple', open_with=default_open,
          mkdirs=default_mkdirs, has_nulls=None, write_index=None,
          partition_on=[], fixed_text=None, append=False,
          object_encoding='infer', times='int64',
          quantization_level=None, quantization_random_seed=None):
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
    has_nulls: None, bool or list of strings
        Whether columns can have nulls. If a list of strings, those given
        columns will be marked as "optional" in the metadata, and include
        null definition blocks on disk. Some data types (floats and times)
        can instead use the sentinel values NaN and NaT, which are not the same
        as NULL in parquet, but functionally act the same in many cases,
        particularly if converting back to pandas later. A value of None
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
        be encoded to bytes. Possible values are bytes|utf8|json|bson, where
        bytes is assumed if not specified (i.e., no conversion). The special
        value 'infer' will cause the type to be guessed from the first ten
        values.
    times: 'int64' (default), or 'int96':
        In "int64" mode, datetimes are written as 8-byte integers, us
        resolution; in "int96" mode, they are written as 12-byte blocks, with
        the first 8 bytes as ns within the day, the next 4 bytes the julian day.
        'int96' mode is included only for compatibility.
    quantization_level: int or None
        If not None, then quantization_level floating point values in this many units of the
        median absolute deviation of the column. Only works for the 'simple' file
        scheme.
    quantization_random_seed: int (greater than 0), `np.random.RandomState`, or None
        The seed or RNG for the dithering of the quantized float values. If None, then the
        a `np.random.RandomState` instance is instantiated using its default (non-deterministic)
        seeding.

    Examples
    --------
    >>> fastparquet.write('myfile.parquet', df)  # doctest: +SKIP
    """

    global COLUMN_MADS
    COLUMN_MADS = {}

    sep = sep_from_open(open_with)
    if isinstance(row_group_offsets, int):
        l = len(data)
        nparts = max((l - 1) // row_group_offsets + 1, 1)
        chunksize = max(min((l - 1) // nparts + 1, l), 1)
        row_group_offsets = list(range(0, l, chunksize))
    if write_index or write_index is None and index_like(data.index):
        data = data.reset_index()
    check_column_names(data.columns, partition_on, fixed_text, object_encoding,
                       has_nulls)
    ignore = partition_on if file_scheme != 'simple' else []
    fmd = make_metadata(data, has_nulls=has_nulls, ignore_columns=ignore,
                        fixed_text=fixed_text, object_encoding=object_encoding,
                        times=times)

    if quantization_level:
        if isinstance(quantization_random_seed, np.random.RandomState):
            quantization_random_state = quantization_random_seed
        else:
            quantization_random_state = np.random.RandomState(seed=quantization_random_seed)
    else:
        quantization_random_state = None

    if file_scheme == 'simple':
        write_simple(filename, data, fmd, row_group_offsets,
                     compression, open_with, has_nulls, append,
                     quantization_level=quantization_level,
                     quantization_random_state=quantization_random_state)
    elif file_scheme == 'hive':
        if append:
            pf = api.ParquetFile(filename, open_with=open_with)
            if pf.file_scheme != 'hive':
                raise ValueError('Requested file scheme is hive, '
                                 'but existing file scheme is not.')
            fmd = pf.fmd
            i_offset = find_max_part(fmd.row_groups)
            partition_on = list(pf.cats)
        else:
            i_offset = 0
        fn = sep.join([filename, '_metadata'])
        mkdirs(filename)
        for i, start in enumerate(row_group_offsets):
            end = (row_group_offsets[i+1] if i < (len(row_group_offsets) - 1)
                   else None)
            part = 'part.%i.parquet' % (i + i_offset)
            if partition_on:
                partition_on_columns(
                    data[start:end], partition_on, filename, part, fmd,
                    sep, compression, open_with, mkdirs
                )
                rg = None
            else:
                partname = sep.join([filename, part])
                with open_with(partname, 'wb') as f2:
                    rg = make_part_file(f2, data[start:end], fmd.schema,
                                        compression=compression)
                for chunk in rg.columns:
                    chunk.file_path = part

            if rg is not None:
                fmd.row_groups.append(rg)

        write_common_metadata(fn, fmd, open_with, no_row_groups=False)
        write_common_metadata(sep.join([filename, '_common_metadata']), fmd,
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


def partition_on_columns(data, columns, root_path, partname, fmd, sep,
                         compression, open_with, mkdirs):
    """
    Split each row-group by the given columns

    Each combination of column values (determined by pandas groupby) will
    be written in structured directories.
    """
    gb = data.groupby(columns)
    remaining = list(data)
    for column in columns:
        remaining.remove(column)
    for key in gb.indices:
        df = gb.get_group(key)[remaining]
        path = sep.join(["%s=%s" % (name, val)
                         for name, val in zip(columns, key)])
        relname = sep.join([path, partname])
        mkdirs(root_path + sep + path)
        fullname = sep.join([root_path, path, partname])
        with open_with(fullname, 'wb') as f2:
            rg = make_part_file(f2, df, fmd.schema,
                                compression=compression)
        if rg is not None:
            for chunk in rg.columns:
                chunk.file_path = relname
            fmd.row_groups.append(rg)


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


def merge(file_list, verify_schema=True, open_with=default_open):
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

    Returns
    -------
    ParquetFile instance corresponding to the merged data.
    """
    sep = sep_from_open(open_with)
    basepath, fmd = metadata_from_many(file_list, verify_schema, open_with)

    out_file = sep.join([basepath, '_metadata'])
    write_common_metadata(out_file, fmd, open_with, no_row_groups=False)
    out = api.ParquetFile(out_file, open_with=open_with)

    out_file = sep.join([basepath, '_common_metadata'])
    write_common_metadata(out_file, fmd, open_with)
    return out

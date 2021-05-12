from copy import copy
import json
import re
import struct
import warnings
from bisect import bisect

import numpy as np
import pandas as pd
from pandas.core.arrays.masked import BaseMaskedDtype

from fastparquet.util import join_path
from .thrift_structures import write_thrift

from pandas.api.types import is_categorical_dtype
from .thrift_structures import parquet_thrift
from .compression import compress_data
from .converted_types import tobson
from . import encoding, api, __version__
from .util import (default_open, default_mkdirs,
                   check_column_names, metadata_from_many, created_by,
                   get_column_metadata, path_string)
from .speedups import array_encode_utf8, pack_byte_array
from . import cencoding
from .cencoding import NumpyIO
from decimal import Decimal

MARKER = b'PAR1'
NaT = np.timedelta64(None).tobytes()  # require numpy version >= 1.7
nat = np.datetime64('NaT').view('int64')

typemap = {  # primitive type, converted type, bit width
    'boolean': (parquet_thrift.Type.BOOLEAN, None, 1),
    'Int32': (parquet_thrift.Type.INT32, None, 32),
    'Int64': (parquet_thrift.Type.INT64, None, 64),
    'Int8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_8, 8),
    'Int16': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.INT_16, 16),
    'UInt8': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_8, 8),
    'UInt16': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_16, 16),
    'UInt32': (parquet_thrift.Type.INT32, parquet_thrift.ConvertedType.UINT_32, 32),
    'UInt64': (parquet_thrift.Type.INT64, parquet_thrift.ConvertedType.UINT_64, 64),
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

pdoptional_to_numpy_typemap = {
    pd.Int8Dtype(): np.int8,
    pd.Int16Dtype(): np.int16,
    pd.Int32Dtype(): np.int32,
    pd.Int64Dtype(): np.int64,
    pd.UInt8Dtype(): np.uint8,
    pd.UInt16Dtype(): np.uint16,
    pd.UInt32Dtype(): np.uint32,
    pd.UInt64Dtype(): np.uint64,
    pd.BooleanDtype(): bool
}


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
        elif object_encoding == 'decimal':
            type, converted_type, width = (parquet_thrift.Type.DOUBLE, None,
                                           64)
        else:
            raise ValueError('Object encoding (%s) not one of '
                             'infer|utf8|bytes|json|bson|bool|int|int32|float|decimal' %
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
            warnings.warn(
                'Coercing datetimes to UTC before writing the parquet file, the timezone is stored in the metadata. '
                'Reading back with fastparquet/pyarrow will restore the timezone properly.'
            )
    elif dtype.kind == "m":
        type, converted_type, width = (parquet_thrift.Type.INT64,
                                       parquet_thrift.ConvertedType.TIME_MICROS, None)
    elif str(dtype) == 'string':
        if object_encoding == 'infer':
            object_encoding = infer_object_encoding(data)

        if object_encoding == 'utf8':
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                           parquet_thrift.ConvertedType.UTF8,
                                           None)
        elif object_encoding in ['bytes', None]:
            type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY, None,
                                           None)
        else:
            raise ValueError('Object Encoding (%s) not one of infer|utf8|bytes' % object_encoding)
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
            # TODO: with our own bitpack writer, no need to copy for
            #  the padding
            padded = np.lib.pad(data.values, (0, 8 - (len(data) % 8)),
                                'constant', constant_values=(0, 0))
            out = np.packbits(padded.reshape(-1, 8)[:, ::-1].ravel())
        elif dtype.name in typemap:
            out = data.values
    elif "S" in str(dtype)[:2] or "U" in str(dtype)[:2]:
        out = data.values
    elif dtype == "O":
        # TODO: nullable types
        try:
            if converted_type == parquet_thrift.ConvertedType.UTF8:
                # getattr for new pandas StringArray
                # TODO: to bytes in one step
                out = array_encode_utf8(data)
            elif converted_type == parquet_thrift.ConvertedType.DECIMAL:
                out = data.values.astype(np.float64, copy=False)
            elif converted_type is None:
                if type in revmap:
                    out = data.values.astype(revmap[type], copy=False)
                elif type == parquet_thrift.Type.BOOLEAN:
                    # TODO: with our own bitpack writer, no need to copy for
                    #  the padding
                    padded = np.lib.pad(data.values, (0, 8 - (len(data) % 8)),
                                        'constant', constant_values=(0, 0))
                    out = np.packbits(padded.reshape(-1, 8)[:, ::-1].ravel())
                else:
                    out = data.values
            elif converted_type == parquet_thrift.ConvertedType.JSON:
                # TODO: avoid list, use better JSON
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
    elif str(dtype) == "string":
        try:
            if converted_type == parquet_thrift.ConvertedType.UTF8:
                # TODO: into bytes in one step
                out = array_encode_utf8(data)
            elif converted_type is None:
                out = data.values
            if type == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
                out = out.astype('S%i' % se.type_length)
        except Exception as e:  # pragma: no cover
            ct = parquet_thrift.ConvertedType._VALUES_TO_NAMES[
                converted_type] if converted_type is not None else None
            raise ValueError('Error converting column "%s" to bytes using '
                             'encoding %s. Original error: '
                             '%s' % (data.name, ct, e))

    elif converted_type == parquet_thrift.ConvertedType.TIMESTAMP_MICROS:
        # TODO: shift inplace
        out = np.empty(len(data), 'int64')
        time_shift(data.values.view('int64'), out)
    elif converted_type == parquet_thrift.ConvertedType.TIME_MICROS:
        # TODO: shift inplace
        out = np.empty(len(data), 'int64')
        time_shift(data.values.view('int64'), out)
    elif type == parquet_thrift.Type.INT96 and dtype.kind == 'M':
        ns_per_day = (24 * 3600 * 1000000000)
        day = data.values.view('int64') // ns_per_day + 2440588
        ns = (data.values.view('int64') % ns_per_day)  # - ns_per_day // 2
        out = np.empty(len(data), dtype=[('ns', 'i8'), ('day', 'i4')])
        out['ns'] = ns
        out['day'] = day
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    return out


def infer_object_encoding(data):
    head = data[:10] if isinstance(data, pd.Index) else data.dropna()[:10]
    if all(isinstance(i, str) for i in head):
        return "utf8"
    elif all(isinstance(i, bytes) for i in head):
        return 'bytes'
    elif all(isinstance(i, (list, dict)) for i in head):
        return 'json'
    elif all(isinstance(i, bool) for i in head):
        return 'bool'
    elif all(isinstance(i, Decimal) for i in head):
        return 'decimal'
    elif all(isinstance(i, int) for i in head):
        return 'int'
    elif all(isinstance(i, float) or isinstance(i, np.floating)
             for i in head):
        # You need np.floating here for pandas NaNs in object
        # columns with python floats.
        return 'float'
    else:
        raise ValueError("Can't infer object conversion type: %s" % head)


def time_shift(indata, outdata, factor=1000):
    outdata.view("int64")[:] = np.where(
        indata.view('int64') == nat,
        nat,
        indata.view('int64') // factor
    )


def encode_plain(data, se):
    """PLAIN encoding; returns byte representation"""
    out = convert(data, se)
    if se.type == parquet_thrift.Type.BYTE_ARRAY:
        return pack_byte_array(list(out))
    else:
        return out.tobytes()


def encode_dict(data, se):
    """ The data part of dictionary encoding is always int8/16, with RLE/bitpack
    """
    width = data.values.dtype.itemsize * 8
    buf = np.empty(10, dtype=np.uint8)
    o = NumpyIO(buf)
    o.write_byte(width)
    bit_packed_count = (len(data) + 7) // 8
    cencoding.encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header
    # TODO: `bytes`, `tobytes` makes copy, and adding bytes also makes copy
    return bytes(o.so_far()) + data.values.tobytes()


encode = {
    'PLAIN': encode_plain,
    'RLE_DICTIONARY': encode_dict,
}


def make_definitions(data, no_nulls):
    """For data that can contain NULLs, produce definition levels binary
    data: either bitpacked bools, or (if number of nulls == 0), single RLE
    block."""
    buf = np.empty(10, dtype=np.uint8)
    temp = NumpyIO(buf)

    if no_nulls:
        # no nulls at all
        l = len(data)
        cencoding.encode_unsigned_varint(l << 1, temp)
        temp.write_byte(1)
        # TODO: adding bytes causes copy
        block = struct.pack('<i', temp.tell()) + temp.so_far()
        out = data
    else:
        se = parquet_thrift.SchemaElement(type=parquet_thrift.Type.BOOLEAN)
        out = encode_plain(data.notnull(), se)

        cencoding.encode_unsigned_varint(len(out) << 1 | 1, temp)
        head = temp.so_far()

        # TODO: adding bytes causes copy
        block = struct.pack('<i', len(head) + len(out)) + head + out
        out = data.dropna()  # better, data[data.notnull()], from above ?
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
    compression: str, dict, or None
        if ``str``, must be one of the keys in ``compression.compress``
        if ``dict``, must have key ``"type"`` which specifies the compression
        type to use, which must be one of the keys in ``compression.compress``,
        and may optionally have key ``"args`` which should be a dictionary of
        options to pass to the underlying compression engine.

    Returns
    -------
    chunk: ColumnChunk structure

    """
    has_nulls = selement.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    tot_rows = len(data)
    encoding = "PLAIN"

    if has_nulls:
        if is_categorical_dtype(data.dtype):
            num_nulls = (data.cat.codes == -1).sum()
        else:
            num_nulls = len(data) - data.count()
        definition_data, data = make_definitions(data, num_nulls == 0)
        # make_definitions returns `data` with all nulls dropped
        # the null-stripped `data` can be converted from Optional Types to
        # their numpy counterparts
        if isinstance(data.dtype, BaseMaskedDtype):
            data = data.astype(pdoptional_to_numpy_typemap[data.dtype])
        if data.dtype.kind == "O" and not is_categorical_dtype(data.dtype):
            try:
                if selement.type == parquet_thrift.Type.INT64:
                    data = data.astype(int)
                elif selement.type == parquet_thrift.Type.BOOLEAN:
                    data = data.astype(bool)
            except ValueError as e:
                t = parquet_thrift.Type._VALUES_TO_NAMES[selement.type]
                raise ValueError('Error converting column "%s" to primitive '
                                 'type %s. Original error: '
                                 '%s' % (data.name, t, e))

    else:
        definition_data = b""
        num_nulls = 0

    # No nested field handling (encode those as J/BSON)
    repetition_data = b""

    cats = False
    name = data.name
    diff = 0
    max, min = None, None
    start = f.tell()

    if is_categorical_dtype(data.dtype):
        dph = parquet_thrift.DictionaryPageHeader(
                num_values=len(data.cat.categories),
                encoding=parquet_thrift.Encoding.PLAIN)
        bdata = encode['PLAIN'](pd.Series(data.cat.categories), selement)
        # TODO: copy on bytes addition
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
                if selement.type == parquet_thrift.Type.BYTE_ARRAY:
                    if selement.converted_type is not None:
                        max = encode['PLAIN'](pd.Series([max]), selement)[4:]
                        min = encode['PLAIN'](pd.Series([min]), selement)[4:]
                else:
                    max = encode['PLAIN'](pd.Series([max]), selement)
                    min = encode['PLAIN'](pd.Series([min]), selement)
        except TypeError:
            pass
        ncats = len(data.cat.categories)
        data = data.cat.codes
        cats = True
        encoding = "RLE_DICTIONARY"
    elif str(data.dtype) in ['int8', 'int16', 'uint8', 'uint16']:
        # encoding = "RLE"
        # disallow bitpacking for compatability
        data = data.astype('int32')

    # TODO: here we make a copy of encoded data even if deps
    #  and reps are empty
    bdata = definition_data + repetition_data + encode[encoding](
            data, selement)
    # here we make another copy
    bdata += 8 * b'\x00'
    try:
        if encoding != 'RLE_DICTIONARY' and num_nulls == 0:
            max, min = data.values.max(), data.values.min()
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

    if cats:
        p = [
            parquet_thrift.PageEncodingStats(
                page_type=parquet_thrift.PageType.DICTIONARY_PAGE,
                encoding=parquet_thrift.Encoding.PLAIN, count=1),
            parquet_thrift.PageEncodingStats(
                page_type=parquet_thrift.PageType.DATA_PAGE,
                encoding=parquet_thrift.Encoding.RLE_DICTIONARY, count=1),
        ]
        encodings = [parquet_thrift.Encoding.PLAIN,
                     parquet_thrift.Encoding.RLE_DICTIONARY]

    else:
        p = [parquet_thrift.PageEncodingStats(
             page_type=parquet_thrift.PageType.DATA_PAGE,
             encoding=parquet_thrift.Encoding.PLAIN, count=1)]
        encodings = [parquet_thrift.Encoding.PLAIN]

    if isinstance(compression, dict):
        algorithm = compression.get("type", None)
    else:
        algorithm = compression

    cmd = parquet_thrift.ColumnMetaData(
            type=selement.type, path_in_schema=[name],
            encodings=encodings,
            codec=(getattr(parquet_thrift.CompressionCodec, algorithm.upper())
                   if algorithm else 0),
            num_values=tot_rows,
            statistics=s,
            data_page_offset=start,
            encoding_stats=p,
            key_value_metadata=[],
            total_uncompressed_size=uncompressed_size,
            total_compressed_size=compressed_size)
    if cats:
        cmd.dictionary_page_offset = dict_start
        cmd.key_value_metadata.append(
            parquet_thrift.KeyValue(key='num_categories', value=str(ncats)))
        cmd.key_value_metadata.append(
            parquet_thrift.KeyValue(key='numpy_dtype', value=str(data.dtype)))
    chunk = parquet_thrift.ColumnChunk(file_offset=offset,
                                       meta_data=cmd)
    write_thrift(f, chunk)
    return chunk


def make_row_group(f, data, schema, compression=None):
    """ Make a single row group of a Parquet file """
    rows = len(data)
    if rows == 0:
        return
    if any(not isinstance(c, (bytes, str)) for c in data):
        raise ValueError('Column names must be str or bytes:',
                         {c: type(c) for c in data.columns
                          if not isinstance(c, (bytes, str))})
    rg = parquet_thrift.RowGroup(num_rows=rows, total_byte_size=0, columns=[])

    for column in schema:
        if column.type is not None:
            if isinstance(compression, dict):
                comp = compression.get(column.name, None)
                if comp is None:
                    comp = compression.get('_default', None)
            else:
                comp = compression
            chunk = write_column(f, data[column.name], column,
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
            fmd = parquet_thrift.FileMetaData(num_rows=rg.num_rows,
                                              schema=schema,
                                              version=1,
                                              created_by=created_by,
                                              row_groups=[rg],)
            foot_size = write_thrift(f, fmd)
            f.write(struct.pack(b"<i", foot_size))
        else:
            fmd = copy(fmd)
            fmd.row_groups = [rg]
            fmd.num_rows = rg.num_rows
            foot_size = write_thrift(f, fmd)
            f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)
    return rg


def make_metadata(data, has_nulls=True, ignore_columns=None, fixed_text=None,
                  object_encoding=None, times='int64', index_cols=None, partition_cols=None):
    if ignore_columns is None:
        ignore_columns = []
    if index_cols is None:
        index_cols = []
    if partition_cols is None:
        partition_cols = []
    if not data.columns.is_unique:
        raise ValueError('Cannot create parquet dataset with duplicate'
                         ' column names (%s)' % data.columns)
    if not isinstance(index_cols, list):
        start = index_cols.start
        stop = index_cols.stop
        step = index_cols.step

        index_cols = [{'name': index_cols.name,
                       'start': start,
                       'stop': stop,
                       'step': step,
                       'kind': 'range'}]
    pandas_metadata = {'index_columns': index_cols,
                       'partition_columns': [],
                       'columns': [],
                       'column_indexes': [{'name': data.columns.name,
                                           'field_name': data.columns.name,
                                           'pandas_type': 'mixed-integer',
                                           'numpy_type': 'object',
                                           'metadata': None}],
                       'creator': {'library': 'fastparquet',
                                   'version': __version__},
                       'pandas_version': pd.__version__,}
    root = parquet_thrift.SchemaElement(name='schema',
                                        num_children=0)

    meta = parquet_thrift.KeyValue()
    meta.key = 'pandas'
    fmd = parquet_thrift.FileMetaData(num_rows=len(data),
                                      schema=[root],
                                      version=1,
                                      created_by=created_by,
                                      row_groups=[],
                                      key_value_metadata=[meta])

    object_encoding = object_encoding or {}
    for column in partition_cols:
        pandas_metadata['partition_columns'].append(get_column_metadata(data[column], column))
    for column in data.columns:
        if column in ignore_columns:
            continue
        pandas_metadata['columns'].append(
            get_column_metadata(data[column], column))
        oencoding = (object_encoding if isinstance(object_encoding, str)
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
        fmd.schema.append(se)
        root.num_children += 1
    meta.value = json.dumps(pandas_metadata, sort_keys=True)
    return fmd


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
          object_encoding='infer', times='int64',
          custom_metadata=None):
    """ Write Pandas DataFrame to filename as Parquet Format.

    Parameters
    ----------
    filename: string
        Parquet collection to write to, either a single file (if file_scheme
        is simple) or a directory containing the metadata and data-files.
    data: pandas dataframe
        The table to write.
    row_group_offsets: int or list of ints
        If int, row-groups will be approximately this many rows, rounded down
        to make row groups about the same size; if a list, the explicit index
        values to start new row groups.
    compression: str, dict
        compression to apply to each column, e.g. ``GZIP`` or ``SNAPPY`` or a
        ``dict`` like ``{"col1": "SNAPPY", "col2": None}`` to specify per
        column compression types.
        In both cases, the compressor settings would be the underlying
        compressor defaults. To pass arguments to the underlying compressor,
        each ``dict`` entry should itself be a dictionary::

            {
                col1: {
                    "type": "LZ4",
                    "args": {
                        "mode": "high_compression",
                        "compression": 9
                     }
                },
                col2: {
                    "type": "SNAPPY",
                    "args": None
                }
                "_default": {
                    "type": "GZIP",
                    "args": None
                }
            }

        where ``"type"`` specifies the compression type to use, and ``"args"``
        specifies a ``dict`` that will be turned into keyword arguments for
        the compressor.
        If the dictionary contains a "_default" entry, this will be used for any
        columns not explicitly specified in the dictionary.
    file_scheme: 'simple'|'hive'|'drill'
        If simple: all goes in a single file
        If hive or drill: each row group is in a separate file, and a separate
        file (called "_metadata") contains the metadata.
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
    append: bool (False) or 'overwrite'
        - If False, construct data-set from scratch; if True, add new row-group(s)
          to existing data-set. In the latter case, the data-set must exist,
          and the schema must match the input data.
        - If 'overwrite', existing partitions will be replaced in-place, where
          the given data has any rows within a given partition. To enable this,
          these other parameters have to be set to specific values, or will
          raise ValueError:
           * ``row_group_offsets=0``
           * ``file_scheme='hive'``
           * ``partition_on`` has to be used, set to at least a column name
    object_encoding: str or {col: type}
        For object columns, this gives the data type, so that the values can
        be encoded to bytes. Possible values are bytes|utf8|json|bson|bool|int|int32|decimal,
        where bytes is assumed if not specified (i.e., no conversion). The
        special value 'infer' will cause the type to be guessed from the first
        ten non-null values. The decimal.Decimal type is a valid choice, but will
        result in float encoding with possible loss of accuracy.
    times: 'int64' (default), or 'int96':
        In "int64" mode, datetimes are written as 8-byte integers, us
        resolution; in "int96" mode, they are written as 12-byte blocks, with
        the first 8 bytes as ns within the day, the next 4 bytes the julian day.
        'int96' mode is included only for compatibility.
    custom_metadata: dict
        key-value metadata to write

    Examples
    --------
    >>> fastparquet.write('myfile.parquet', df)  # doctest: +SKIP
    """
    if str(has_nulls) == 'infer':
        has_nulls = None
    if isinstance(row_group_offsets, int):
        if not row_group_offsets:
            row_group_offsets = [0]
        else:
            l = len(data)
            nparts = max((l - 1) // row_group_offsets + 1, 1)
            chunksize = max(min((l - 1) // nparts + 1, l), 1)
            row_group_offsets = list(range(0, l, chunksize))
    if (write_index or write_index is None
            and not isinstance(data.index, pd.RangeIndex)):
        cols = set(data)
        data = data.reset_index()
        index_cols = [c for c in data if c not in cols]
    elif write_index is None and isinstance(data.index, pd.RangeIndex):
        # write_index=None, range to metadata
        index_cols = data.index
    else:  # write_index=False
        index_cols = []
    check_column_names(data.columns, partition_on, fixed_text, object_encoding,
                       has_nulls)
    ignore = partition_on if file_scheme != 'simple' else []
    fmd = make_metadata(data, has_nulls=has_nulls, ignore_columns=ignore,
                        fixed_text=fixed_text, object_encoding=object_encoding,
                        times=times, index_cols=index_cols,
                        partition_cols=partition_on)
    if custom_metadata is not None:
        fmd.key_value_metadata.extend(
            [
                parquet_thrift.KeyValue(key=key, value=value)
                for key, value in custom_metadata.items()
            ]
        )

    if file_scheme == 'simple':
        write_simple(filename, data, fmd, row_group_offsets,
                     compression, open_with, has_nulls, append)
    elif file_scheme in ['hive', 'drill']:
        if append: # can be True or 'overwrite'
            pf = api.ParquetFile(filename, open_with=open_with)
            if pf.file_scheme not in ['hive', 'empty', 'flat']:
                raise ValueError('Requested file scheme is %s, but '
                                 'existing file scheme is not.' % file_scheme)
            fmd = pf.fmd
            if tuple(partition_on) != tuple(pf.cats):
                raise ValueError('When appending, partitioning columns must'
                                 ' match existing data')
            if append == 'overwrite' and partition_on:
                # Build list of 'path' from existing files
                # (to have partition values).
                exist_rgps = ['_'.join(rg.columns[0].file_path.split('/')[:-1])
                              for rg in fmd.row_groups]
                if len(exist_rgps) > len(set(exist_rgps)):
                    # Some groups are in the same folder (partition). This case
                    # is not handled.
                    raise ValueError("Some partition folders contain several \
part files. This situation is not allowed with use of `append='overwrite'`.")
                i_offset = 0
            else:
                i_offset = find_max_part(fmd.row_groups)
        else:
            i_offset = 0

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
                if append != 'overwrite':
                    # Append or 'standard' write mode.
                    fmd.row_groups.extend(rgs)
                else:
                    # 'overwrite' mode -> update fmd in place.
                    # Get 'new' combinations of values from columns listed in
                    # 'partition_on',along with corresponding row groups.
                    new_rgps = {'_'.join(rg.columns[0].file_path.split('/')[:-1]): rg \
                              for rg in rgs}
                    for part_val in new_rgps:
                        if part_val in exist_rgps:
                            # Replace existing row group metadata with new ones.
                            row_group_index = exist_rgps.index(part_val)
                            fmd.row_groups[row_group_index] = new_rgps[part_val]
                        else:
                            # Insert new rg metadata among existing ones,
                            # preserving order, if the existing list is sorted
                            # in the 1st place.
                            row_group_index = bisect(exist_rgps, part_val)
                            fmd.row_groups.insert(row_group_index, new_rgps[part_val])
                            # Keep 'exist_rgps' list representative for next 'replace'
                            # or 'insert' cases.
                            exist_rgps.insert(row_group_index, part_val)

            else:
                partname = join_path(filename, part)
                with open_with(partname, 'wb') as f2:
                    rg = make_part_file(f2, data[start:end], fmd.schema,
                                        compression=compression, fmd=fmd)
                for chunk in rg.columns:
                    chunk.file_path = part
                fmd.row_groups.append(rg)

        fmd.num_rows = sum(rg.num_rows for rg in fmd.row_groups)
        fn = join_path(filename, '_metadata')
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
    s = re.compile(r'.*part.(?P<i>[\d]+).parquet$')
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
    for key, group in sorted(gb):
        if group.empty:
            continue
        df = group[remaining]
        if not isinstance(key, tuple):
            key = (key,)
        if with_field:
            path = join_path(*(
                "%s=%s" % (name, path_string(val))
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
            fmd = copy(fmd)
            fmd.row_groups = []
            foot_size = write_thrift(f, fmd)
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
                    ncats = [k.value for k in (col.meta_data.key_value_metadata or [])
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

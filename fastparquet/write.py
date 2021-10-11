from bisect import bisect
from copy import copy
import json
import os
import re
import struct

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.core.arrays.masked import BaseMaskedDtype

from . import cencoding
from .cencoding import NumpyIO
from .compression import compress_data
from .converted_types import tobson
from .speedups import array_encode_utf8, pack_byte_array
from .thrift_structures import parquet_thrift, write_thrift
from .util import (created_by, default_open, default_mkdirs, join_path,
                   path_string)

MARKER = b'PAR1'
DATAPAGE_VERSION = 2 if os.environ.get("FASTPARQUET_DATAPAGE_V2", False) else 1

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
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)


def time_shift(indata, outdata, factor=1000):
    outdata.view("int64")[:] = np.where(
        indata.view('int64') == nat,
        nat,
        indata.view('int64') // factor
    )


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
    elif dtype.kind == "M":
        out = data.values.view("int64")
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    return out


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


def make_definitions(data, no_nulls, datapage_version=1):
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
        if datapage_version == 1:
            # TODO: adding bytes causes copy
            block = struct.pack('<i', temp.tell()) + temp.so_far()
        else:
            block = bytes(temp.so_far())
        out = data
    else:
        se = parquet_thrift.SchemaElement(type=parquet_thrift.Type.BOOLEAN)
        out = encode_plain(data.notnull(), se)

        cencoding.encode_unsigned_varint(len(out) << 1 | 1, temp)
        head = temp.so_far()

        # TODO: adding bytes causes copy
        if datapage_version == 1:
            block = struct.pack('<i', len(head) + len(out)) + head + out
        else:
            # no need to write length, it's in the header
            # head.write(out)?
            block = bytes(head) + out
        out = data.dropna()  # better, data[data.notnull()], from above ?
    return block, out


def write_column(f, data, selement, compression=None, datapage_version=None,
                 stats=True):
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
    stats: bool
        Whether to calculate and write summary statistics

    Returns
    -------
    chunk: ColumnChunk structure

    """
    datapage_version = datapage_version or DATAPAGE_VERSION
    has_nulls = selement.repetition_type == parquet_thrift.FieldRepetitionType.OPTIONAL
    tot_rows = len(data)
    encoding = "PLAIN"

    if has_nulls:
        if is_categorical_dtype(data.dtype):
            num_nulls = (data.cat.codes == -1).sum()
        else:
            num_nulls = len(data) - data.count()
        definition_data, data = make_definitions(data, num_nulls == 0, datapage_version=datapage_version)
        # make_definitions returns `data` with all nulls dropped
        # the null-stripped `data` can be converted from Optional Types to
        # their numpy counterparts
        if isinstance(data.dtype, BaseMaskedDtype):
            data = data.astype(pdoptional_to_numpy_typemap[data.dtype])
        if data.dtype.kind == "O" and not is_categorical_dtype(data.dtype):
            try:
                if selement.type in [parquet_thrift.Type.INT64,
                                     parquet_thrift.Type.INT32]:
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
        l0 = len(bdata)
        if compression and compression.upper() != "UNCOMPRESSED":
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
            if stats:
                # TODO: this max/min works, but is slow
                max, min = np.array(data[data.notnull()]).max(), np.array(data[data.notnull()]).min()
                if selement.type == parquet_thrift.Type.BYTE_ARRAY:
                    if selement.converted_type is not None:
                        max = encode['PLAIN'](pd.Series([max]), selement)[4:]
                        min = encode['PLAIN'](pd.Series([min]), selement)[4:]
                else:
                    max = encode['PLAIN'](pd.Series([max]), selement)
                    min = encode['PLAIN'](pd.Series([min]), selement)
        except (TypeError, ValueError):
            pass
        ncats = len(data.cat.categories)
        data = data.cat.codes
        cats = True
        encoding = "RLE_DICTIONARY"
    elif str(data.dtype) in ['int8', 'int16', 'uint8', 'uint16']:
        # encoding = "RLE"
        # disallow bit-packing for compatibility
        data = data.astype('int32')

    try:
        if encoding != 'RLE_DICTIONARY':
            # for categorical, we already did this above
            if stats:
                max, min = data[data.notnull()].values.max(), data[data.notnull()].values.min()
                if selement.type == parquet_thrift.Type.BYTE_ARRAY:
                    if selement.converted_type is not None:
                        # max = max.encode("utf8") ?
                        max = encode['PLAIN'](pd.Series([max], name=data.name), selement)[4:]
                        min = encode['PLAIN'](pd.Series([min], name=data.name), selement)[4:]
                else:
                    max = encode['PLAIN'](pd.Series([max], name=data.name), selement)
                    min = encode['PLAIN'](pd.Series([min], name=data.name), selement)
    except (TypeError, ValueError):
        pass
    s = parquet_thrift.Statistics(max=max, min=min, null_count=num_nulls) if stats else None

    if datapage_version == 1:
        bdata = b"".join([
            repetition_data, definition_data, encode[encoding](data, selement), 8 * b'\x00'
        ])
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
    elif datapage_version == 2:
        is_compressed = isinstance(compression, dict) or (
            compression is not None and compression.upper() != "UNCOMPRESSED")
        dph = parquet_thrift.DataPageHeaderV2(
            num_values=tot_rows,
            num_nulls=num_nulls,
            num_rows=tot_rows,
            encoding=getattr(parquet_thrift.Encoding, encoding),
            definition_levels_byte_length=len(definition_data),
            repetition_levels_byte_length=0,  # len(repetition_data),
            is_compressed=is_compressed,
            statistics=s
        )
        bdata = encode[encoding](data, selement)
        lb = len(bdata)
        if is_compressed:
            bdata = compress_data(bdata, compression)
            diff = lb - len(bdata)
        else:
            diff = 0
        ph = parquet_thrift.PageHeader(type=parquet_thrift.PageType.DATA_PAGE_V2,
                                       uncompressed_page_size=lb + len(definition_data),
                                       compressed_page_size=len(bdata) + len(definition_data),
                                       data_page_header_v2=dph, crc=None)
        write_thrift(f, ph)
        # f.write(repetition_data)  # no-op
        f.write(definition_data)
        f.write(bdata)

    compressed_size = f.tell() - start
    uncompressed_size = compressed_size + diff

    offset = f.tell()

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
    return chunk


def make_row_group(f, data, schema, compression=None, stats=True):
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
            st = stats if isinstance(stats, bool) else column.name in stats
            chunk = write_column(f, data[column.name], column,
                                 compression=comp, stats=st)
            rg.columns.append(chunk)
    rg.total_byte_size = sum([c.meta_data.total_uncompressed_size for c in
                              rg.columns])
    return rg


def make_part_file(f, data, schema, compression=None, fmd=None,
                   stats=True):
    if len(data) == 0:
        return
    with f as f:
        f.write(MARKER)
        rg = make_row_group(f, data, schema, compression=compression,
                            stats=stats)
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


def partition_on_columns(data, columns, root_path, partname, fmd,
                         compression, open_with, mkdirs, with_field=True,
                         stats=True):
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
                                compression=compression, fmd=fmd, stats=stats)
        if rg is not None:
            for chunk in rg.columns:
                chunk.file_path = relname
            rgs.append(rg)
    return rgs


def write_simple(fn, data, fmd, row_group_offsets, compression,
                 open_with=default_open, append=False, stats=True):
    """
    Write to one single parquet file (for file_scheme='simple').
    """
    mode = 'rb+' if append else 'wb'

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
                                compression=compression, stats=stats)
            if rg is not None:
                fmd.row_groups.append(rg)

        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)


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


def write_multi(fn, data, fmd, row_group_offsets, compression, file_scheme,
                open_with=default_open, mkdirs=default_mkdirs, partition_on=[],
                append=False, stats=True, write_fmd=True):
    """
    Write to multi parquet files (for file_scheme='hive', 'drill' or 'flat').
    """
    if not append:
        # New dataset.
        i_offset = 0
        mkdirs(fn)
    elif append is True:
        i_offset = find_max_part(fmd.row_groups)
    else:
        # 'overwrite'.
        i_offset = 0
        exist_rgps = [rg.columns[0].file_path.rsplit('/',1)[0]
                      for rg in fmd.row_groups]

    for i, start in enumerate(row_group_offsets):
        end = (row_group_offsets[i+1] if i < (len(row_group_offsets) - 1)
               else None)
        part = 'part.%i.parquet' % (i + i_offset)
        if partition_on:
            rgs = partition_on_columns(data[start:end], partition_on, fn, part,
                                       fmd,  compression, open_with, mkdirs,
                                       with_field=file_scheme == 'hive',
                                       stats=stats)
            if append != 'overwrite':
                # Append or 'standard' write mode.
                fmd.row_groups.extend(rgs)
            else:
                # 'overwrite' mode -> update fmd in place.
                # Get 'new' combinations of values from columns listed in
                # 'partition_on',along with corresponding row groups.
                new_rgps = {rg.columns[0].file_path.rsplit('/',1)[0]: rg \
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
                        fmd.row_groups.insert(row_group_index,
                                              new_rgps[part_val])
                        # Keep 'exist_paths' list representative for next
                        # 'replace' or 'insert' cases.
                        exist_rgps.insert(row_group_index, part_val)

        else:
            partname = join_path(fn, part)
            with open_with(partname, 'wb') as f2:
                rg = make_part_file(f2, data[start:end], fmd.schema,
                                    compression=compression, fmd=fmd, stats=stats)
            for chunk in rg.columns:
                chunk.file_path = part
            fmd.row_groups.append(rg)

    fmd.num_rows = sum(rg.num_rows for rg in fmd.row_groups)
    if write_fmd:
        write_common_metadata(join_path(fn, '_metadata'), fmd, open_with,
                              no_row_groups=False)
        write_common_metadata(join_path(fn, '_common_metadata'), fmd,
                              open_with)

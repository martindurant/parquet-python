from __future__ import print_function

import json
import struct

import numpy as np
import pandas as pd
from six import integer_types

from fastparquet import ParquetFile
from fastparquet.compression import compress_data
from fastparquet.converted_types import tobson
from fastparquet.encoding import width_from_max_int, Encoder
from fastparquet.schema import NUMPY_OBJECT, NUMPY_INTEGER, NUMPY_BOOLEAN, NUMPY_DATETIME
from fastparquet.speedups import array_encode_utf8
from fastparquet.thrift_structures import parquet_thrift, write_thrift
from fastparquet.util import default_open, default_mkdirs, index_like, PY2, STR_TYPE, check_column_names, join_path, created_by
from fastparquet.writer import is_categorical_dtype, encode, find_max_part, partition_on_columns, make_part_file, write_common_metadata, typemap, revmap, time_shift, MARKER
from mo_dots import split_field
from mo_parquet import SchemaTree, rows_to_columns


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
    elif dtype == NUMPY_OBJECT:
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
                               dtype=NUMPY_OBJECT)
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
    elif type == parquet_thrift.Type.INT96 and dtype.kind == NUMPY_DATETIME:
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

    if is_categorical_dtype(data.dtype):
        num_nulls = (data.cat.codes == -1).sum()
    elif data.dtype.kind in [NUMPY_INTEGER, NUMPY_BOOLEAN]:
        num_nulls = 0
    else:
        num_nulls = len(data) - data.values.count()

    if data.dtype.kind == NUMPY_OBJECT and not is_categorical_dtype(data.dtype):
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
    name = data.name
    diff = 0
    max, min = None, None

    if is_categorical_dtype(data.dtype):
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
    elif str(data.dtype) in ['int8', 'int16', 'uint8', 'uint16']:
        encoding = "RLE"

    start = f.tell()
    bdata = Encoder()
    bit_width = width_from_max_int(data.max_definition_level)
    bdata.rle_bit_packed_hybrid(data.reps, bit_width)
    bdata.rle_bit_packed_hybrid(data.defs, bit_width)
    bdata.bytes(encode[encoding](data, selement))
    bdata.zeros(8)

    try:
        if encoding != 'PLAIN_DICTIONARY' and num_nulls == 0:
            max, min = np.max(data.values), np.min(data.values)
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
    f.write(bdata.to_bytearay())

    compressed_size = f.tell() - start
    uncompressed_size = compressed_size + diff

    offset = f.tell()
    s = parquet_thrift.Statistics(max=max, min=min, null_count=num_nulls)

    p = [parquet_thrift.PageEncodingStats(
        page_type=parquet_thrift.PageType.DATA_PAGE,
        encoding=parquet_thrift.Encoding.PLAIN, count=1)]

    cmd = parquet_thrift.ColumnMetaData(
        type=selement.type,
        path_in_schema=split_field(name),
        encodings=[
            parquet_thrift.Encoding.RLE,
            parquet_thrift.Encoding.BIT_PACKED,
            parquet_thrift.Encoding.PLAIN
        ],
        codec=(
            getattr(parquet_thrift.CompressionCodec, compression.upper())
            if compression else 0
        ),
        num_values=tot_rows,
        statistics=s,
        data_page_offset=start,
        encoding_stats=p,
        key_value_metadata=[],
        total_uncompressed_size=uncompressed_size,
        total_compressed_size=compressed_size
    )
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
            chunk = write_column(
                f,
                data.get_column(column.name),
                column,
                compression=comp
            )
            rg.columns.append(chunk)
    rg.total_byte_size = sum([c.meta_data.total_uncompressed_size for c in
                              rg.columns])
    return rg


def write_simple(fn, data, fmd, row_group_offsets, compression,
                 open_with, has_nulls, append=False):
    """
    Write to one single file (for file_scheme='simple')
    """
    if append:
        pf = ParquetFile(fn, open_with=open_with)
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
            f.seek(-(head_size + 8), 2)
        else:
            f.write(MARKER)
        for i, start in enumerate(row_group_offsets):
            end = (row_group_offsets[i + 1] if i < (len(row_group_offsets) - 1)
                   else None)
            rg = make_row_group(
                f,
                data[start:end],
                fmd.schema,
                compression=compression
            )
            if rg is not None:
                fmd.row_groups.append(rg)

        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)


def write(filename, rows, row_group_offsets=50000000,
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
    data = rows_to_columns(rows, schema)

    if str(has_nulls) == 'infer':
        has_nulls = None
    if isinstance(row_group_offsets, int):
        l = len(data)
        nparts = max((l - 1) // row_group_offsets + 1, 1)
        chunksize = max(min((l - 1) // nparts + 1, l), 1)
        row_group_offsets = list(range(0, l, chunksize))
    index_cols = []
    check_column_names(data.columns, partition_on, fixed_text, object_encoding,
                       has_nulls)
    ignore = partition_on if file_scheme != 'simple' else []

    children = data.schema.get_parquet_metadata()
    fmd = parquet_thrift.FileMetaData(
        num_rows=len(data),
        schema=[parquet_thrift.SchemaElement(name='.', num_children=len(children))] + children,
        version=1,
        created_by=created_by,
        row_groups=[],
        key_value_metadata=[]
    )

    if file_scheme == 'simple':
        write_simple(filename, data, fmd, row_group_offsets,
                     compression, open_with, has_nulls, append)
    elif file_scheme in ['hive', 'drill']:
        if append:
            pf = ParquetFile(filename, open_with=open_with)
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
            end = (row_group_offsets[i + 1] if i < (len(row_group_offsets) - 1)
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



import json
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype

from .thrift_structures import parquet_thrift, write_thrift
from . import encoding, api, __version__
from .util import (check_column_names, created_by, default_open,
                   default_mkdirs, get_column_metadata)
from .write import (consolidate_categories, DATAPAGE_VERSION, make_definitions,
                    make_part_file, make_row_group, MARKER,
                    partition_on_columns, row_idx_to_cols, typemap,
                    write_common_metadata, write_multi, write_simple)
from . import cencoding
from .cencoding import NumpyIO
from decimal import Decimal

NaT = np.timedelta64(None).tobytes()  # require numpy version >= 1.7


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
    logical_type = None
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
            # output will have the same resolution as original data, for resolution <= ms
            tz = getattr(dtype, "tz", None) is not None
            if "ns" in dtype.str:
                type = parquet_thrift.Type.INT64
                converted_type = None
                logical_type = parquet_thrift.LogicalType(
                    TIMESTAMP=parquet_thrift.TimestampType(
                        isAdjustedToUTC=tz,
                        unit=parquet_thrift.TimeUnit(NANOS=parquet_thrift.NanoSeconds())
                    )
                )
                width = None
            elif "us" in dtype.str:
                type, converted_type, width = (
                    parquet_thrift.Type.INT64,
                    parquet_thrift.ConvertedType.TIMESTAMP_MICROS, None
                )
                logical_type = parquet_thrift.LogicalType(
                    TIMESTAMP=parquet_thrift.TimestampType(
                        isAdjustedToUTC=tz,
                        unit=parquet_thrift.TimeUnit(MICROS=parquet_thrift.MicroSeconds())
                    )
                )

            else:
                type, converted_type, width = (
                    parquet_thrift.Type.INT64,
                    parquet_thrift.ConvertedType.TIMESTAMP_MILLIS, None
                )
                logical_type = parquet_thrift.LogicalType(
                    TIMESTAMP=parquet_thrift.TimestampType(
                        isAdjustedToUTC=tz,
                        unit=parquet_thrift.TimeUnit(MILLIS=parquet_thrift.MilliSeconds())
                    )
                )
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
    elif "string" in str(dtype):
        type, converted_type, width = (parquet_thrift.Type.BYTE_ARRAY,
                                       parquet_thrift.ConvertedType.UTF8,
                                       None)
    else:
        raise ValueError("Don't know how to convert data type: %s" % dtype)
    se = parquet_thrift.SchemaElement(
        name=data.name, type_length=width,
        converted_type=converted_type, type=type,
        repetition_type=parquet_thrift.FieldRepetitionType.REQUIRED,
        logicalType=logical_type
    )
    return se, type


def infer_object_encoding(data):
    head = data[:10] if isinstance(data, pd.Index) else data.dropna()[:10]
    if all(isinstance(i, str) for i in head if i is not None):
        return "utf8"
    elif all(isinstance(i, bytes) for i in head if i is not None):
        return 'bytes'
    elif all(isinstance(i, (list, dict)) for i in head if i is not None):
        return 'json'
    elif all(isinstance(i, bool) for i in head if i is not None):
        return 'bool'
    elif all(isinstance(i, Decimal) for i in head if i is not None):
        return 'decimal'
    elif all(isinstance(i, int) for i in head if i is not None):
        return 'int'
    elif all(isinstance(i, float) or isinstance(i, np.floating)
             for i in head if i):
        # You need np.floating here for pandas NaNs in object
        # columns with python floats.
        return 'float'
    else:
        raise ValueError("Can't infer object conversion type: %s" % head)


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


def write(filename, data, row_group_offsets=50000000,
          compression=None, file_scheme='simple', open_with=default_open,
          mkdirs=default_mkdirs, has_nulls=True, write_index=None,
          partition_on=[], fixed_text=None, append=False,
          object_encoding='infer', times='int64',
          custom_metadata=None, stats=True):
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
        to make row groups about the same size;
        If a list, the explicit index values to start new row groups.
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
        If the dictionary contains a "_default" entry, this will be used for
        any columns not explicitly specified in the dictionary.
    file_scheme: 'simple'|'hive'|'drill'
        If simple: all goes in a single file;
        If hive or drill: each row group is in a separate file, and a separate
        file (called "_metadata") contains the metadata.
    open_with: function
        When called with a f(path, mode), returns an open file-like object.
    mkdirs: function
        When called with a path/URL, creates any necessary dictionaries to
        make that location writable, e.g., ``os.makedirs``. This is not
        necessary if using the simple file scheme.
    has_nulls: bool, 'infer' or list of strings
        Whether columns can have nulls. If a list of strings, those given
        columns will be marked as "optional" in the metadata, and include
        null definition blocks on disk. Some data types (floats and times)
        can instead use the sentinel values NaN and NaT, which are not the same
        as NULL in parquet, but functionally act the same in many cases,
        particularly if converting back to pandas later. A value of 'infer'
        will assume nulls for object columns and not otherwise.
        Ignored if appending to an existing parquet data-set.
    write_index: boolean
        Whether or not to write the index to a separate column.  By default we
        write the index *if* it is not 0, 1, ..., n.
        Ignored if appending to an existing parquet data-set.
    partition_on: list of column names
        Passed to groupby in order to split data within each row-group,
        producing a structured directory tree. Note: as with pandas, null
        values will be dropped. Ignored if file_scheme is simple.
        Checked when appending to an existing parquet dataset that requested
        partition column names match those of existing parquet data-set.
    fixed_text: {column: int length} or None
        For bytes or str columns, values will be converted
        to fixed-length strings of the given length for the given columns
        before writing, potentially providing a large speed
        boost. The length applies to the binary representation *after*
        conversion for utf8, json or bson.
        Ignored if appending to an existing parquet dataset.
    append: bool (False) or 'overwrite'
        If False, construct data-set from scratch;
        If True, add new row-group(s) to existing data-set. In the latter case,
        the data-set must exist, and the schema must match the input data;

        If 'overwrite', existing partitions will be replaced in-place, where
        new data has any rows within a given partition. To enable this,
        following parameters have to be set to specific values, or will raise
        ValueError:

           *  ``row_group_offsets=0``
           *  ``file_scheme='hive'``
           *  ``partition_on`` has to be used, set to at least a column name

    object_encoding: str or {col: type}
        For object columns, this gives the data type, so that the values can
        be encoded to bytes. Possible values are
        bytes|utf8|json|bson|bool|int|int32|decimal, where bytes is assumed if
        not specified (i.e., no conversion). The special value 'infer' will
        cause the type to be guessed from the first ten non-null values. The
        decimal.Decimal type is a valid choice, but will result in float
        encoding with possible loss of accuracy.
        Ignored if appending to an existing parquet data-set.
    times: 'int64' (default), or 'int96':
        In "int64" mode, datetimes are written as 8-byte integers, us
        resolution; in "int96" mode, they are written as 12-byte blocks, with
        the first 8 bytes as ns within the day, the next 4 bytes the julian day.
        'int96' mode is included only for compatibility.
        Ignored if appending to an existing parquet data-set.
    custom_metadata: dict
        Key-value metadata to write.
        Ignored if appending to an existing parquet data-set.
    stats: True|False|list(str)
        Whether to calculate and write summary statistics. If True (default), do it for
        every column; if False, never do; and if a list of str, do it only for those
        specified columns.

    Examples
    --------
    >>> fastparquet.write('myfile.parquet', df)  # doctest: +SKIP
    """
    if file_scheme not in ('simple', 'hive', 'drill'):
        raise ValueError('File scheme should be simple|hive, not ',
                         file_scheme)
    # Express 'row_group_offsets' as a list of row indexes.
    # TODO: wrap this 'if' branch' into a separate function, and extend it so
    # that instead of a target number of rows per row group, it accepts a
    # target size per row group, and define from it chunks of data.
    if isinstance(row_group_offsets, int):
        if not row_group_offsets:
            row_group_offsets = [0]
        else:
            l = len(data)
            nparts = max((l - 1) // row_group_offsets + 1, 1)
            chunksize = max(min((l - 1) // nparts + 1, l), 1)
            row_group_offsets = list(range(0, l, chunksize))
    if append:
        # Case 'append=True' or 'overwrite'.
        pf = api.ParquetFile(filename, open_with=open_with)
        if file_scheme == 'simple':
            # Case 'simple'
            if pf.file_scheme not in ['simple', 'empty']:
                raise ValueError('File scheme requested is simple, but '
                                 'existing file scheme is %s.' % pf.file_scheme)
        else:
            # Case 'hive', 'drill'
            if pf.file_scheme not in ['hive', 'empty', 'flat']:
                raise ValueError('Requested file scheme is %s, but '
                                 'existing file scheme is not.' % file_scheme)
            if tuple(partition_on) != tuple(pf.cats):
                raise ValueError('When appending, partitioning columns must'
                                 ' match existing data')
        pf.append_as_row_groups(data, row_group_offsets, compression,
                                open_with, mkdirs, append, stats,
                                write_fmd=True)
    else:
        # Case 'append=False'.
        # Initialize common metadata.
        # Define 'index_cols' to be recorded in metadata.
        if (write_index or write_index is None
                and not isinstance(data.index, pd.RangeIndex)):
            cols = set(data)
            data = row_idx_to_cols(data)
            index_cols = [c for c in data if c not in cols]
        elif write_index is None and isinstance(data.index, pd.RangeIndex):
            # write_index=None, range to metadata
            index_cols = data.index
        else:
            # write_index=False
            index_cols = []
        if str(has_nulls) == 'infer':
            has_nulls = None
        check_column_names(data.columns, partition_on, fixed_text,
                           object_encoding, has_nulls)
        ignore = partition_on if file_scheme != 'simple' else []
        fmd = make_metadata(data, has_nulls=has_nulls, ignore_columns=ignore,
                            fixed_text=fixed_text,
                            object_encoding=object_encoding,
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
            # Case 'simple'
            write_simple(filename, data, fmd, row_group_offsets,
                         compression, open_with, append=False, stats=stats)
        else:
            # Case 'hive', 'drill'
            write_multi(filename, data, fmd, row_group_offsets, compression,
                        file_scheme, open_with, mkdirs, partition_on,
                        append=False, stats=stats)


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
    out = api.ParquetFile(file_list, verify_schema, open_with, root)
    out._write_common_metadata(open_with, update_num_rows=False)
    return out

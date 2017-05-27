import ast
import os, os.path
import shutil
import pandas as pd
import pytest
import re
import tempfile
import thriftpy
import sys
import six


PY2 = six.PY2
PY3 = six.PY3
STR_TYPE = six.string_types[0]  # 'str' for Python3, 'basestring' for Python2
created_by = "fastparquet-python version 1.0.0 (build 111)"


class ParquetException(Exception):
    """Generic Exception related to unexpected data format when
     reading parquet file."""
    pass


def sep_from_open(opener):
    if opener is default_open:
        return os.sep
    else:
        return '/'


if PY2:
    def default_mkdirs(f):
        if not os.path.exists(f):
            os.makedirs(f)
else:
    def default_mkdirs(f):
        os.makedirs(f, exist_ok=True)


def default_open(f, mode='rb'):
    return open(f, mode)


def val_to_num(x):
    if x in ['NOW', 'TODAY']:
        return x
    try:
        return ast.literal_eval(x)
    except:
        pass
    try:
        return pd.to_datetime(x)
    except:
        pass
    try:
        return pd.to_timedelta(x)
    except:
        return x


@pytest.yield_fixture()
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)

if PY2:
    def ensure_bytes(s):
        return s.encode('utf-8') if isinstance(s, unicode) else s
else:
    def ensure_bytes(s):
        return s.encode('utf-8') if isinstance(s, str) else s


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


def thrift_copy(structure):
    """
    Recursively copy a thriftpy structure
    """
    base = structure.__class__()
    for key in dir(structure):
        if key.startswith('_') or key in ['thrift_spec', 'read', 'write',
                                          'default_spec']:
            continue
        val = getattr(structure, key)
        if isinstance(val, list):
            setattr(base, key, [thrift_copy(item)
                                if isinstance(item, thriftpy.thrift.TPayload)
                                else item for item in val])
        elif isinstance(val, thriftpy.thrift.TPayload):
            setattr(base, key, thrift_copy(val))
        else:
            setattr(base, key, val)
    return base


def index_like(index):
    """
    Does index look like a default range index?
    """
    return not (isinstance(index, pd.RangeIndex) and
                index._start == 0 and
                index._stop == len(index) and
                index._step == 1 and index.name is None)


def check_column_names(columns, *args):
    """Ensure that parameters listing column names have corresponding columns"""
    for arg in args:
        if isinstance(arg, (tuple, list)):
            if set(arg) - set(columns):
                raise ValueError("Column name not in list.\n"
                                 "Requested %s\n"
                                 "Allowed %s" % (arg, columns))


def byte_buffer(raw_bytes):
    return buffer(raw_bytes) if PY2 else memoryview(raw_bytes)


def metadata_from_many(file_list, verify_schema=False, open_with=default_open):
    """
    Given list of parquet files, make a FileMetaData that points to them

    Parameters
    ----------
    file_list: list of paths of parquet files
    verify_schema: bool (False)
        Whether to assert that the schemas in each file are identical
    open_with: function
        Use this to open each path.

    Returns
    -------
    basepath: the root path that other paths are relative to
    fmd: metadata thrift structure
    """
    from fastparquet import api
    sep = sep_from_open(open_with)
    if all(isinstance(pf, api.ParquetFile) for pf in file_list):
        pfs = file_list
        file_list = [pf.fn for pf in pfs]
    elif all(not isinstance(pf, api.ParquetFile) for pf in file_list):
        pfs = [api.ParquetFile(fn, open_with=open_with) for fn in file_list]
    else:
        raise ValueError("Merge requires all PaquetFile instances or none")
    basepath, file_list = analyse_paths(file_list, sep)

    if verify_schema:
        for pf in pfs[1:]:
            if pf._schema != pfs[0]._schema:
                raise ValueError('Incompatible schemas')

    fmd = thrift_copy(pfs[0].fmd)  # we inherit "created by" field
    fmd.row_groups = []

    for pf, fn in zip(pfs, file_list):
        if pf.file_scheme != 'simple':
            raise ValueError('Cannot merge multi-file input', fn)
        for rg in pf.row_groups:
            rg = thrift_copy(rg)
            for chunk in rg.columns:
                chunk.file_path = fn
            fmd.row_groups.append(rg)

    fmd.num_rows = sum(rg.num_rows for rg in fmd.row_groups)
    return basepath, fmd

# simple cache to avoid re compile every time
seps = {}


def ex_from_sep(sep):
    """Generate regex for category folder matching"""
    if sep not in seps:
        if sep in r'\^$.|?*+()[]':
            s = re.compile(r"([a-zA-Z_]+)=([^\{}]+)".format(sep))
        else:
            s = re.compile("([a-zA-Z_]+)=([^{}]+)".format(sep))
        seps[sep] = s
    return seps[sep]


def analyse_paths(file_list, sep=os.sep):
    """Consolidate list of file-paths into acceptable parquet relative paths"""
    path_parts_list = [fn.split(sep) for fn in file_list]
    if len({len(path_parts) for path_parts in path_parts_list}) > 1:
        raise ValueError('Mixed nesting in merge files')
    basepath = path_parts_list[0][:-1]
    s = ex_from_sep(sep)
    out_list = []
    for i, path_parts in enumerate(path_parts_list):
        j = len(path_parts) - 1
        for k, (base_part, path_part) in enumerate(zip(basepath, path_parts)):
            if base_part != path_part:
                j = k
                break
        basepath = basepath[:j]
    l = len(basepath)
    if len({tuple([p.split('=')[0] for p in parts[l:-1]])
            for parts in path_parts_list}) > 1:
        raise ValueError('Partitioning directories do not agree')
    for path_parts in path_parts_list:
        for path_part in path_parts[l:-1]:
            if s.match(path_part) is None:
                raise ValueError('Malformed paths set at', sep.join(path_parts))
        out_list.append(sep.join(path_parts[l:]))

    return sep.join(basepath), out_list


def infer_dtype(column):
    try:
        return pd.api.types.infer_dtype(column)
    except AttributeError:
        return pd.lib.infer_dtype(column)


def get_column_metadata(column, name):
    """Produce pandas column metadata block"""
    # from pyarrow.pandas_compat
    # https://github.com/apache/arrow/blob/master/python/pyarrow/pandas_compat.py
    inferred_dtype = infer_dtype(column)
    dtype = column.dtype

    if str(dtype) == 'category':
        extra_metadata = {
            'num_categories': len(column.cat.categories),
            'ordered': column.cat.ordered,
        }
        dtype = column.cat.codes.dtype
    elif hasattr(dtype, 'tz'):
        extra_metadata = {'timezone': str(dtype.tz)}
    else:
        extra_metadata = None

    if not isinstance(name, six.string_types):
        raise TypeError(
            'Column name must be a string. Got column {} of type {}'.format(
                name, type(name).__name__
            )
        )

    return {
        'name': name,
        'pandas_type': {
            'string': 'bytes' if PY2 else 'unicode',
            'datetime64': (
                'datetimetz' if hasattr(dtype, 'tz')
                else 'datetime'
            ),
            'integer': str(dtype),
            'floating': str(dtype),
        }.get(inferred_dtype, inferred_dtype),
        'numpy_dtype': str(dtype),
        'metadata': extra_metadata,
    }

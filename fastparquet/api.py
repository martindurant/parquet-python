"""parquet - read parquet files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator
import os
import re
import struct

import numpy as np
import pandas as pd
import thriftpy

from .core import read_thrift
from .thrift_structures import parquet_thrift
from .writer import write
from . import core, schema, converted_types, encoding
from .util import (default_open, ParquetException, sep_from_open, val_to_num,
        ensure_bytes)


class ParquetFile(object):
    """The metadata of a parquet file or collection

    Reads the metadata (row-groups and schema definition) and provides
    methods to extract the data from the files.

    Parameters
    ----------
    fn: path/URL string
        Location of the data. If a directory, will attempt to read a file
        "_metadata" within that directory.
    verify: bool [False]
        test file start/end byte markers
    open_with: function
        With the signature `func(path, mode)`, returns a context which
        evaluated to a file open for reading. Defaults to the built-in `open`.
    sep: string [`os.sep`]
        Path separator to use, if data is in multiple files.
    """
    def __init__(self, fn, verify=False, open_with=default_open,
                 sep=os.sep):
        try:
            fn2 = sep.join([fn, '_metadata'])
            f = open_with(fn2, 'rb')
            fn = fn2
        except (IOError, OSError):
            f = open_with(fn, 'rb')
        self.open = open_with
        self.fn = fn
        self.sep = sep
        with f as f:
            self._parse_header(f, verify)
        self._read_partitions()

    def _parse_header(self, f, verify=True):
        try:
            f.seek(0)
            if verify:
                assert f.read(4) == b'PAR1'
            f.seek(-8, 2)
            head_size = struct.unpack('<i', f.read(4))[0]
            if verify:
                assert f.read() == b'PAR1'
        except (AssertionError, struct.error):
            raise ParquetException('File parse failed: %s' % self.fn)

        f.seek(-(head_size+8), 2)
        try:
            fmd = read_thrift(f, parquet_thrift.FileMetaData)
        except thriftpy.transport.TTransportException:
            raise ParquetException('Metadata parse failed: %s' %
                                         self.fn)
        self.fmd = fmd
        self.head_size = head_size
        self.version = fmd.version
        self.schema = fmd.schema
        self.row_groups = fmd.row_groups or []
        self.key_value_metadata = fmd.key_value_metadata
        self.created_by = fmd.created_by
        self.group_files = {}
        for i, rg in enumerate(self.row_groups):
            for chunk in rg.columns:
                self.group_files.setdefault(i, set()).add(chunk.file_path)
        self.helper = schema.SchemaHelper(self.schema)

    @property
    def columns(self):
        """ Column names """
        return [f.name for f in self.schema if f.num_children is None]

    @property
    def statistics(self):
        return statistics(self)

    def _read_partitions(self):
        cats = {}
        for rg in self.row_groups:
            for col in rg.columns:
                partitions = re.findall("([a-zA-Z_]+)=([^/]+)/",
                                        col.file_path or "")
                for key, val in partitions:
                    cats.setdefault(key, set()).add(val)
        self.cats = {key: list(v) for key, v in cats.items()}

    def read_row_group_file(self, rg, columns, categories):
        """ Open file for reading, and process it as a row-group """
        ofname = self.sep.join([os.path.dirname(self.fn),
                                rg.columns[0].file_path])
        with self.open(ofname, 'rb') as f:
            return self.read_row_group(rg, columns, categories, infile=f)

    def grab_cats(self, columns, group=0):
        """ Read dictionaries of first row_group

        Used to correctly create metadata for categorical dask dataframes.
        Could be used to check that the same dictionary is used throughout
        the data.

        Parameters
        ----------
        columns: list
            Column names to load
        group: int (0)
            Row group to load

        Returns
        -------
        {column: [list, of, values]}
        """
        rg = self.row_groups[group]
        ofname = self.sep.join([os.path.dirname(self.fn),
                                rg.columns[0].file_path])
        out = {}

        with self.open(ofname, 'rb') as f:
            for column in rg.columns:
                name = ".".join(column.meta_data.path_in_schema)
                if name not in columns:
                    continue
                out[name] = core.read_col(column, self.helper, f,
                                          grab_dict=True)
        return out

    def read_row_group(self, rg, columns, categories, infile=None):
        """
        Access row-group in a file and read some columns into a data-frame.
        """
        out = {}

        for column in rg.columns:
            name = ".".join(column.meta_data.path_in_schema)
            if name not in columns:
                continue

            use = name in categories if categories is not None else False
            s = core.read_col(column, self.helper, infile, use_cat=use)
            out[name] = s
        out = pd.DataFrame(out, columns=columns)

        # apply categories
        for cat in self.cats:
            # *Hard assumption*: all chunks in a row group have the
            # same partition (correct for spark/hive)
            partitions = re.findall("([a-zA-Z_]+)=([^/]+)/",
                                    rg.columns[0].file_path)
            val = [p[1] for p in partitions if p[0] == cat][0]
            codes = np.empty(rg.num_rows, dtype=np.int16)
            codes[:] = self.cats[cat].index(val)
            out[cat] = pd.Categorical.from_codes(
                    codes, [val_to_num(c) for c in self.cats[cat]])
        return out

    def to_pandas(self, columns=None, categories=None, filters=[]):
        """
        Read data from parquet into a Pandas dataframe.

        Parameters
        ----------
        columns: list of names or `None`
            Column to load (see `ParquetFile.columns`). Any columns in the
            data not in this list will be ignored. If `None`, read all columns.
        categories: list of names or `None`
            If a column is encoded using dictionary encoding in every row-group
            and its name is also in this list, it will generate a Pandas
            Category-type column, potentially saving memory and time.
        filters: list of tuples
            Filter syntax: [(column, op, val), ...],
            where op is [==, >, >=, <, <=, !=, in, not in]

        Returns
        -------
        Pandas data-frame
        """
        columns = columns or self.columns
        rgs = [rg for rg in self.row_groups if
               not(filter_out_stats(rg, filters, self.helper)) and
               not(filter_out_cats(rg, filters))]
        if all(column.file_path is None for rg in self.row_groups
               for column in rg.columns):
            with self.open(self.fn) as f:
                tot = [self.read_row_group(rg, columns, categories, infile=f)
                       for rg in rgs]
        else:
            tot = [self.read_row_group_file(rg, columns, categories)
                   for rg in rgs]

        if len(tot) == 0:
            return pd.DataFrame(columns=columns + list(self.cats))

        # TODO: if categories vary from one rg to next, need
        # pandas.types.concat.union_categoricals
        return pd.concat(tot, ignore_index=True)

    @property
    def count(self):
        """ Total number of rows """
        return sum(rg.num_rows for rg in self.row_groups)

    @property
    def info(self):
        """ Some metadata details """
        return {'name': self.fn, 'columns': self.columns,
                'categories': list(self.cats), 'rows': self.count}

    @property
    def dtypes(self):
        """ Implied types of the columns in the schema """
        dtype = {f.name: converted_types.typemap(f)
                 for f in self.schema if f.num_children is None}
        for cat in self.cats:
            dtype[cat] = "category"
            # pd.Series(self.cats[cat]).map(val_to_num).dtype
        return dtype

    def __str__(self):
        return "<Parquet File: %s>" % self.info

    __repr__ = __str__


def filter_out_stats(rg, filters, helper):
    """
    According to the filters, should this row-group be excluded

    Considers the statistics included in the metadata of this row-group

    Parameters
    ----------
    rg: thrift RowGroup structure
    filters: list of 3-tuples
        Structure of each tuple: (column, op, value) where op is one of
        ['==', '!=', '<', '<=', '>', '>=', 'in', 'not in'] and value is
        appropriate for the column in question

    Returns
    -------
    True or False
    """
    if len(filters) == 0:
        return False
    for column in rg.columns:
        vmax, vmin = None, None
        name = ".".join(column.meta_data.path_in_schema)
        app_filters = [f[1:] for f in filters if f[0] == name]
        for op, val in app_filters:
            se = helper.schema_element(name)
            if column.meta_data.statistics is not None:
                s = column.meta_data.statistics
                if s.max is not None:
                    b = s.max if isinstance(s.max, bytes) else bytes(
                            s.max, 'ascii')
                    vmax = encoding.read_plain(b, column.meta_data.type, 1)
                    if se.converted_type:
                        vmax = converted_types.convert(vmax, se)
                if s.min is not None:
                    b = s.min if isinstance(s.min, bytes) else bytes(
                            s.min, 'ascii')
                    vmin = encoding.read_plain(b, column.meta_data.type, 1)
                    if se.converted_type:
                        vmin = converted_types.convert(vmin, se)
                out = filter_val(op, val, vmin, vmax)
                if out is True:
                    return True
    return False


def statistics(obj):
    """
    Return per-column statistics for a ParquerFile

    Parameters
    ----------
    obj: ParquetFile

    Returns
    -------
    dictionary mapping stats (min, max, distinct_count, null_count) to column
    names to lists of values.  ``None``s used if no statistics found.

    Examples
    --------
    >>> statistics(my_parquet_file)
    {'min': {'x': [1, 4], 'y': [5, 3]},
     'max': {'x': [2, 6], 'y': [8, 6]},
     'distinct_count': {'x': [None, None], 'y': [None, None]},
     'null_count': {'x': [0, 3], 'y': [0, 0]}}
    """
    if isinstance(obj, parquet_thrift.ColumnChunk):
        md = obj.meta_data
        s = obj.meta_data.statistics
        rv = {}
        if not s:
            return rv
        if s.max is not None:
            rv['max'] = encoding.read_plain(ensure_bytes(s.max), md.type, 1)[0]
        if s.min is not None:
            rv['min'] = encoding.read_plain(ensure_bytes(s.min), md.type, 1)[0]
        if s.null_count is not None:
            rv['null_count'] = s.null_count
        if s.distinct_count is not None:
            rv['distinct_count'] = s.distinct_count
        return rv

    if isinstance(obj, parquet_thrift.RowGroup):
        return {'.'.join(c.meta_data.path_in_schema): statistics(c)
                for c in obj.columns}

    if isinstance(obj, ParquetFile):
        L = list(map(statistics, obj.row_groups))
        names = obj.columns
        d = {n: {col: [item[col].get(n, None) for item in L]
                 for col in obj.columns}
             for n in ['min', 'max', 'null_count', 'distinct_count']}
        helper = schema.SchemaHelper(obj.schema)
        for col in obj.row_groups[0].columns:
            column = '.'.join(col.meta_data.path_in_schema)
            se = helper.schema_element(column)
            if se.converted_type:
                for name in ['min', 'max']:
                    d[name][column] = [x if x is None else converted_types.convert(x, se)
                                       for x in d[name][column]]
        return d


def sorted_partitioned_columns(pf):
    """
    The columns that are known to be sorted partition-by-partition

    They may not be sorted within each partition, but all elements in one
    row group are strictly greater than all elements in previous row groups.

    Examples
    --------
    >>> sorted_partitioned_columns(pf)
    {'id': {'min': [1, 5, 10], 'max': [4, 9, 20]}}

    Returns
    -------
    A set of column names

    See Also
    --------
    statistics
    """
    s = statistics(pf)
    columns = pf.columns
    out = dict()
    for c in columns:
        min, max = s['min'][c], s['max'][c]
        if any(x is None for x in min + max):
            continue
        if (sorted(min) == min and
            sorted(max) == max and
            all(mx < mn for mx, mn in zip(max[:-1], min[1:]))):
            out[c] = {'min': min, 'max': max}
    return out



def filter_out_cats(rg, filters):
    """
    According to the filters, should this row-group be excluded

    Considers the partitioning category applicable to this row-group

    Parameters
    ----------
    rg: thrift RowGroup structure
    filters: list of 3-tuples
        Structure of each tuple: (column, op, value) where op is one of
        ['==', '!=', '<', '<=', '>', '>=', 'in', 'not in'] and value is
        appropriate for the column in question

    Returns
    -------
    True or False
    """
    if len(filters) == 0:
        return False
    partitions = re.findall("([a-zA-Z_]+)=([^/]+)/",
                            rg.columns[0].file_path)
    pairs = [(p[0], val_to_num(p[1])) for p in partitions]
    for cat, v in pairs:

        app_filters = [f[1:] for f in filters if f[0] == cat]
        for op, val in app_filters:
            out = filter_val(op, val, v, v)
            if out is True:
                return True
    return False


def filter_val(op, val, vmin=None, vmax=None):
    """
    Perform value comparison for filtering

    op: ['==', '!=', '<', '<=', '>', '>=', 'in', 'not in']
    val: appropriate value
    vmin, vmax: the range to compare within

    Returns
    -------
    True or False
    """
    if vmin is not None:
        if op in ['==', '>='] and val > vmax:
            return True
        if op == '>' and val >= vmax:
            return True
        if op == 'in' and min(val) > vmax:
            return True
    if vmax is not None:
        if op in ['==', '<='] and val < vmin:
            return True
        if op == '<' and val <= vmin:
            return True
        if op == 'in' and max(val) < vmin:
            return True
    if (op == '!=' and vmax is not None and vmin is not None and
            vmax == vmin and val == vmax):
        return True
    if (op == 'not in' and vmax is not None and vmin is not None and
            vmax == vmin and vmax in val):
        return True

    # keep this row_group
    return False

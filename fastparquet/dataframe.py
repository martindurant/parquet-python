import re
from collections import OrderedDict
from distutils.version import LooseVersion
import numpy as np
import pandas as pd
from pandas.core.index import CategoricalIndex, RangeIndex, Index, MultiIndex
from pandas.core.internals import BlockManager
from pandas import Categorical, DataFrame, Series, __version__ as pdver
from pandas.api.types import is_categorical_dtype
try:
    from itertools import zip_longest
except ImportError:  # For python 2.7
    from itertools import izip_longest as zip_longest
import ast
import operator
import collections
import six
from .util import STR_TYPE


_pandas_logical_type_map = {
    'date': 'datetime64[D]',
    'unicode': np.unicode_,
    'bytes': np.bytes_,
    'string': np.str_,
    'empty': np.object_,
    'mixed': np.object_,
    'mixed-integer': np.object_
}


class Dummy(object):
    pass


def empty(types, size, cats=None, cols=None, index_types=None, index_names=None,
          timezones=None, md=None):
    """
    Create empty DataFrame to assign into

    In the simplest case, will return a Pandas dataframe of the given size,
    with columns of the given names and types. The second return value `views`
    is a dictionary of numpy arrays into which you can assign values that
    show up in the dataframe.

    For categorical columns, you get two views to assign into: if the
    column name is "col", you get both "col" (the category codes) and
    "col-catdef" (the category labels).

    For a single categorical index, you should use the `.set_categories`
    method of the appropriate "-catdef" columns, passing an Index of values

    ``views['index-catdef'].set_categories(pd.Index(newvalues), fastpath=True)``

    Multi-indexes work a lot like categoricals, even if the types of each
    index are not themselves categories, and will also have "-catdef" entries
    in the views. However, these will be Dummy instances, providing only a
    ``.set_categories`` method, to be used as above.

    Parameters
    ----------
    types: like np record structure, 'i4,u2,f4,f2,f4,M8,m8', or using tuples
        applies to non-categorical columns. If there are only categorical
        columns, an empty string of None will do.
    size: int
        Number of rows to allocate
    cats: dict {col: labels}
        Location and labels for categorical columns, e.g., {1: ['mary', 'mo]}
        will create column index 1 (inserted amongst the numerical columns)
        with two possible values. If labels is an integers, `{'col': 5}`,
        will generate temporary labels using range. If None, or column name
        is missing, will assume 16-bit integers (a reasonable default).
    cols: list of labels
        assigned column names, including categorical ones.
    index_types: list of str
        For one of more index columns, make them have this type. See general
        description, above, for caveats about multi-indexing. If None, the
        index will be the default RangeIndex.
    index_names: list of str
        Names of the index column(s), if using
    timezones: dict {col: timezone_str}
        for timestamp type columns, apply this timezone to the pandas series;
        the numpy view will be UTC.
    md: dict
        Pandas metadata dictionary.  Used to reconstruct column index, if present.

    Returns
    -------
    - dataframe with correct shape and data-types
    - list of numpy views, in order, of the columns of the dataframe. Assign
        to this.
    """
    views = {}
    timezones = timezones or {}

    if isinstance(types, STR_TYPE):
        types = types.split(',')
    cols = cols if cols is not None else range(len(types))

    def cat(col):
        if cats is None or col not in cats:
            return RangeIndex(0, 2**14)
        elif isinstance(cats[col], int):
            return RangeIndex(0, cats[col])
        else:  # explicit labels list
            return cats[col]

    df = OrderedDict()
    for t, col in zip(types, cols):
        if str(t) == 'category':
            df[six.text_type(col)] = Categorical([], categories=cat(col),
                                                 fastpath=True)
        else:
            if hasattr(t, 'base'):
                # funky pandas not-dtype
                t = t.base
            d = np.empty(0, dtype=t)
            if d.dtype.kind == "M" and six.text_type(col) in timezones:
                d = Series(d).dt.tz_localize(timezones[six.text_type(col)])
            df[six.text_type(col)] = d

    df = DataFrame(df)
    if not index_types:
        index = RangeIndex(size)
    elif len(index_types) == 1:
        t, col = index_types[0], index_names[0]
        if col is None:
            raise ValueError('If using an index, must give an index name')
        if str(t) == 'category':
            c = Categorical([], categories=cat(col), fastpath=True)
            vals = np.zeros(size, dtype=c.codes.dtype)
            index = CategoricalIndex(c)
            index._data._codes = vals
            views[col] = vals
            views[col+'-catdef'] = index._data
        else:
            d = np.empty(size, dtype=t)
            index = Index(d)
            views[col] = index.values
    else:
        index = MultiIndex([[]], [[]])
        # index = MultiIndex.from_arrays(indexes)
        index._levels = list()
        index._labels = list()
        index._codes = list()
        for i, col in enumerate(index_names):
            index._levels.append(Index([None]))

            def set_cats(values, i=i, col=col, **kwargs):
                values.name = col
                if index._levels[i][0] is None:
                    index._levels[i] = values
                elif not index._levels[i].equals(values):
                    raise RuntimeError("Different dictionaries encountered"
                                       " while building categorical")

            x = Dummy()
            x._set_categories = set_cats

            d = np.zeros(size, dtype=int)
            if LooseVersion(pdver) >= LooseVersion("0.24.0"):
                index._codes = list(index._codes) + [d]
            else:
                index._labels.append(d)
            views[col] = d
            views[col+'-catdef'] = x

    axes = [df._data.axes[0], index]

    # allocate and create blocks
    blocks = []
    for block in df._data.blocks:
        if block.is_categorical:
            categories = block.values.categories
            code = np.zeros(shape=size, dtype=block.values.codes.dtype)
            values = Categorical(values=code, categories=categories,
                                 fastpath=True)
            new_block = block.make_block_same_class(values=values)
        elif getattr(block.dtype, 'tz', None):
            new_shape = (size, )
            values = np.empty(shape=new_shape, dtype='M8[ns]')
            new_block = block.make_block_same_class(
                    values=values, dtype=block.values.dtype)
        else:
            new_shape = (block.values.shape[0], size)
            values = np.empty(shape=new_shape, dtype=block.values.dtype)
            new_block = block.make_block_same_class(values=values)

        blocks.append(new_block)

    # create block manager
    df = DataFrame(BlockManager(blocks, axes))

    # create views
    for block in df._data.blocks:
        dtype = block.dtype
        inds = block.mgr_locs.indexer
        if isinstance(inds, slice):
            inds = list(range(inds.start, inds.stop, inds.step))
        for i, ind in enumerate(inds):
            col = df.columns[ind]
            if is_categorical_dtype(dtype):
                views[col] = block.values._codes
                views[col+'-catdef'] = block.values
            elif getattr(block.dtype, 'tz', None):
                views[col] = np.asarray(block.values, dtype='M8[ns]')
            else:
                views[col] = block.values[i]

    if index_names:
        df.index.names = [
            None if re.match(r'__index_level_\d+__', n) else n
            for n in index_names
        ]

    # Make sure dataframe has correct column index as specified by metdata (#409)
    if md is not None:
        column_index = _construct_column_index(df.columns, md['columns'],
                                               md.get('column_indexes', None))
        df.columns = column_index

    return df, views


def _construct_column_index(column_strings, all_columns, column_indexes):
    """Return pandas index reconstructed from metadata for desired columns

    Parameters
    ----------
    column_strings : list of strings
        Names of desired columns

    all_columns : list of dicts
        Dictionary of column metadata

    column_indexes : list of dicts
        Column_index metadata.not

    Returns
    -------
    columns : pandas index of columns
    """
    # Create default for column_indexes, if not provided
    if column_indexes is None:
        column_indexes = [{"name": None, "field_name": None,
                           "pandas_type": "empty",
                           "numpy_type": "object",
                           "metadata": None}]

    if all_columns:
        columns_name_dict = {
            c.get('field_name', _column_name_to_strings(c['name'])): c['name']
            for c in all_columns
        }
        columns_values = [
            columns_name_dict.get(name, name) for name in column_strings
        ]
    else:
        columns_values = column_strings

    # If we're passed multiple column indexes then evaluate with
    # ast.literal_eval, since the column index values show up as a list of
    # tuples
    to_pair = ast.literal_eval if len(column_indexes) > 1 else lambda x: (x,)

    # Create the column index

    # Construct the base index
    if not isinstance(columns_values, pd.Index) and not columns_values:
        columns = pd.Index(columns_values)
    else:
        columns = pd.MultiIndex.from_tuples(
            list(map(to_pair, columns_values)),
            names=[col_index['name'] for col_index in column_indexes] or None,
        )

    # if we're reconstructing the index
    if len(column_indexes) > 0:
        columns = _reconstruct_columns_from_metadata(columns, column_indexes)

    # ARROW-1751: flatten a single level column MultiIndex for pandas 0.21.0
    columns = _flatten_single_level_multiindex(columns)

    return columns


def _flatten_single_level_multiindex(index):
    if isinstance(index, pd.MultiIndex) and index.nlevels == 1:
        levels, = index.levels
        labels, = _get_multiindex_codes(index)

        # Cheaply check that we do not somehow have duplicate column names
        if not index.is_unique:
            raise ValueError('Found non-unique column index')

        return pd.Index([levels[_label] if _label != -1 else None
                         for _label in labels],
                        name=index.names[0])
    return index


def _reconstruct_columns_from_metadata(columns, column_indexes):
    """Construct a pandas MultiIndex from `columns` and column index metadata
    in `column_indexes`.


    Parameters
    ----------
    columns : List[pd.Index]
        Desired columns
    column_indexes : List[Dict[str, str]]
        The column index metadata.

    Returns
    -------
    result : MultiIndex
        The index reconstructed using `column_indexes` metadata with levels of
        the correct type.

    Notes
    -----
    * adapted from pyarrow/pandas_compat
   """
    # Get levels and labels, and provide sane defaults if the index has a
    # single level to avoid if/else spaghetti.
    levels = getattr(columns, 'levels', None) or [columns]
    labels = _get_multiindex_codes(columns) or [
        pd.RangeIndex(len(level)) for level in levels
    ]

    # Convert each level to the dtype provided in the metadata
    levels_dtypes = [
        (level, col_index.get('pandas_type', str(level.dtype)))
        for level, col_index in zip_longest(
            levels, column_indexes, fillvalue={}
        )
    ]

    new_levels = []
    encoder = operator.methodcaller('encode', 'UTF-8')
    for level, pandas_dtype in levels_dtypes:
        dtype = _pandas_type_to_numpy_type(pandas_dtype)

        # Since our metadata is UTF-8 encoded, Python turns things that were
        # bytes into unicode strings when json.loads-ing them. We need to
        # convert them back to bytes to preserve metadata.
        if dtype == np.bytes_:
            level = level.map(encoder)
        elif level.dtype != dtype:
            level = level.astype(dtype)

        new_levels.append(level)

    return pd.MultiIndex(new_levels, labels, names=columns.names)

def _column_name_to_strings(name):
    """Convert a column name (or level) to either a string or a recursive
    collection of strings.

    Parameters
    ----------
    name : str or tuple

    Returns
    -------
    value : str or tuple

    Examples
    --------
    >>> name = 'foo'
    >>> _column_name_to_strings(name)
    'foo'
    >>> name = ('foo', 'bar')
    >>> _column_name_to_strings(name)
    ('foo', 'bar')
    >>> import pandas as pd
    >>> name = (1, pd.Timestamp('2017-02-01 00:00:00'))
    >>> _column_name_to_strings(name)
    ('1', '2017-02-01 00:00:00')
    """
    if isinstance(name, six.string_types):
        return name
    elif isinstance(name, six.binary_type):
        # XXX: should we assume that bytes in Python 3 are UTF-8?
        return name.decode('utf8')
    elif isinstance(name, tuple):
        return str(tuple(map(_column_name_to_strings, name)))
    elif isinstance(name, collections.Sequence):
        raise TypeError("Unsupported type for MultiIndex level")
    elif name is None:
        return None
    return str(name)


def _pandas_type_to_numpy_type(pandas_type):
    """Get the numpy dtype that corresponds to a pandas type.

    Parameters
    ----------
    pandas_type : str
        The result of a call to pandas.lib.infer_dtype.

    Returns
    -------
    dtype : np.dtype
        The dtype that corresponds to `pandas_type`.
    """
    try:
        return _pandas_logical_type_map[pandas_type]
    except KeyError:
        return np.dtype(pandas_type)

def _get_multiindex_codes(mi):
    # compat for pandas < 0.24 (MI labels renamed to codes).
    if isinstance(mi, pd.MultiIndex):
        return mi.codes if hasattr(mi, 'codes') else mi.labels
    else:
        return None


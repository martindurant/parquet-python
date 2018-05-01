import re
from collections import OrderedDict
import numpy as np
from pandas.core.index import CategoricalIndex, RangeIndex, Index, MultiIndex
from pandas.core.internals import BlockManager
from pandas import Categorical, DataFrame, Series
from pandas.api.types import is_categorical_dtype
from .util import STR_TYPE


def empty(types, size, cats=None, cols=None, index_types=None, index_names=None,
          timezones=None):
    """
    Create empty DataFrame to assign into

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
    timezones: dict {col: timezone_str}
        for timestamp type columns, apply this timezone to the pandas series;
        the numpy view will be UTC.

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
            df[str(col)] = Categorical([], categories=cat(col), fastpath=True)
        else:
            d = np.empty(0, dtype=t)
            if d.dtype.kind == "M" and str(col) in timezones:
                d = Series(d).dt.tz_localize(timezones[str(col)])
            df[str(col)] = d

    df = DataFrame(df)
    if not index_types:
        indexes = [RangeIndex(size)]
    else:
        indexes = []
        for t, col in zip(index_types, index_names):
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
                # if d.dtype.kind == "M" and str(col) in timezones:
                #     d = Series(d).dt.tz_localize(timezones[str(col)])
                index = Index(d)
                views[col] = index.values
            indexes.append(index)

    if len(indexes) > 1:
        levels = [
            views[n] if str(t) != 'category' else views[n+'-catdef'].categories
            for t, n in zip(index_types, index_names)
        ]
        labels = [
            range(size) if str(t) != 'category' else views[n]
            for t, n in zip(index_types, index_names)
        ]
        index = MultiIndex(levels=levels, labels=labels, names=index_names)
    else:
        index = indexes[0]

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
            values = np.empty(shape=new_shape, dtype=block.values.values.dtype)
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
                views[col] = block.values.values
            else:
                views[col] = block.values[i]

    if index_names:
        df.index.names = [
            None if re.match(r'__index_level_\d+__', n) else n
            for n in index_names
        ]
    return df, views

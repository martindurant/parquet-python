from collections import OrderedDict
import numpy as np
from pandas.core.index import CategoricalIndex, RangeIndex, Index
from pandas.core.internals import BlockManager
from pandas import Categorical, DataFrame, Series
from pandas.api.types import is_categorical_dtype
from .util import STR_TYPE


def empty(types, size, cats=None, cols=None, index_type=None, index_name=None,
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
    df = OrderedDict()
    for t, col in zip(types, cols):
        if str(t) == 'category':
            if cats is None or col not in cats:
                df[str(col)] = Categorical(
                        [], categories=RangeIndex(0, 2**14),
                        fastpath=True)
            elif isinstance(cats[col], int):
                df[str(col)] = Categorical(
                        [], categories=RangeIndex(0, cats[col]),
                        fastpath=True)
            else:  # explicit labels list
                df[str(col)] = Categorical([], categories=cats[col],
                                           fastpath=True)
        else:
            d = np.empty(0, dtype=t)
            if d.dtype.kind == "M" and str(col) in timezones:
                d = Series(d).dt.tz_localize(timezones[str(col)])
            df[str(col)] = d
    df = DataFrame(df)

    if index_type is not None and index_type is not False:
        if index_name is None:
            raise ValueError('If using an index, must give an index name')
        if str(index_type) == 'category':
            if cats is None or index_name not in cats:
                c = Categorical(
                        [], categories=RangeIndex(0, 2**14),
                        fastpath=True)
            elif isinstance(cats[index_name], int):
                c = Categorical(
                        [], categories=RangeIndex(0, cats[index_name]),
                        fastpath=True)
            else:  # explicit labels list
                c = Categorical([], categories=cats[index_name],
                                fastpath=True)
            vals = np.empty(size, dtype=c.codes.dtype)
            index = CategoricalIndex(c)
            index._data._codes = vals
            views[index_name] = vals
        else:
            index = Index(np.empty(size, dtype=index_type))
            views[index_name] = index.values

        axes = [df._data.axes[0], index]
    else:
        axes = [df._data.axes[0], RangeIndex(size)]

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

    if index_name is not None and index_name is not False:
        df.index.name = index_name
    if str(index_type) == 'category':
        views[index_name+'-catdef'] = df._data.axes[1].values
    return df, views

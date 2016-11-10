
import os

import numpy as np
import pandas as pd

from fastparquet.util import tempdir
from fastparquet import write, ParquetFile
from fastparquet.api import statistics, sorted_partitioned_columns

TEST_DATA = "test-data"


def test_statistics(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1.0, 2.0, 1.0],
                       'z': ['a', 'b', 'c']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, partitions=[0, 2])

    p = ParquetFile(fn)

    s = statistics(p)
    expected = {'distinct_count': {'x': [None, None],
                                   'y': [None, None],
                                   'z': [None, None]},
                'max': {'x': [2, 3], 'y': [2.0, 1.0], 'z': [b'b', b'c']},
                'min': {'x': [1, 3], 'y': [1.0, 1.0], 'z': [b'a', b'c']},
                'null_count': {'x': [0, 0], 'y': [0, 0], 'z': [0, 0]}}

    assert s == expected


def test_logical_types(tempdir):
    df = pd.util.testing.makeMixedDataFrame()

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, partitions=[0, 2])

    p = ParquetFile(fn)

    s = statistics(p)

    assert isinstance(s['min']['D'][0], np.datetime64)


def test_empty_statistics(tempdir):
    p = ParquetFile(os.path.join(TEST_DATA, "nation.impala.parquet"))

    s = statistics(p)
    assert s == {'distinct_count': {'n_comment': [None],
                                    'n_name': [None],
                                    'n_nationkey': [None],
                                    'n_regionkey': [None]},
                  'max': {'n_comment': [None],
                          'n_name': [None],
                          'n_nationkey': [None],
                          'n_regionkey': [None]},
                  'min': {'n_comment': [None],
                          'n_name': [None],
                          'n_nationkey': [None],
                          'n_regionkey': [None]},
                  'null_count': {'n_comment': [None],
                                 'n_name': [None],
                                 'n_nationkey': [None],
                                 'n_regionkey': [None]}}


def test_sorted_partitioned_columns(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, partitions=[0, 2])

    pf = ParquetFile(fn)

    result = sorted_partitioned_columns(pf)
    expected = {'x': {'min': [1, 3], 'max': [2, 4]},
                'z': {'min': [b'a', b'c'], 'max': [b'b', b'd']}}

    assert result == expected

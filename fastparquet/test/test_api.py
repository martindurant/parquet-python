# -*- coding: utf-8 -*-
import io
import os
import subprocess
import sys
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
try:
    from pandas.tslib import Timestamp
except ImportError:
    from pandas import Timestamp
import pytest

from fastparquet.test.util import tempdir
import fastparquet
from fastparquet import write, ParquetFile
from fastparquet.api import statistics, sorted_partitioned_columns, filter_in, filter_not_in
from fastparquet.util import join_path

TEST_DATA = "test-data"
WIN = os.name == 'nt'


@pytest.mark.xfail(reason="new numpy")
def test_import_without_warning():
    # in a subprocess to avoid import chacing issues.
    subprocess.check_call([sys.executable, "-Werror", "-c", "import fastparquet"])


def test_statistics(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3],
                       'y': [1.0, 2.0, 1.0],
                       'z': ['a', 'b', 'c']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])

    p = ParquetFile(fn)

    s = statistics(p)
    expected = {'distinct_count': {'x': [None, None],
                                   'y': [None, None],
                                   'z': [None, None]},
                'max': {'x': [2, 3], 'y': [2.0, 1.0], 'z': ['b', 'c']},
                'min': {'x': [1, 3], 'y': [1.0, 1.0], 'z': ['a', 'c']},
                'null_count': {'x': [0, 0], 'y': [0, 0], 'z': [0, 0]}}

    assert s == expected


def test_logical_types(tempdir):
    df = pd.util.testing.makeMixedDataFrame()

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])

    p = ParquetFile(fn)

    s = statistics(p)

    assert isinstance(s['min']['D'][0], (np.datetime64, Timestamp))


def test_text_schema(tempdir):
    df = pd.util.testing.makeMixedDataFrame()
    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df)
    p = ParquetFile(fn)
    t = p.schema.text
    expected = ('- schema: \n'
                '| - A: DOUBLE, OPTIONAL\n'
                '| - B: DOUBLE, OPTIONAL\n'
                '| - C: BYTE_ARRAY, UTF8, OPTIONAL\n'
                '  - D: INT64, TIMESTAMP[NANOS], OPTIONAL')
    assert t == expected
    assert repr(p.schema) == "<Parquet Schema with 5 entries>"


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


def test_sorted_row_group_columns(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'v': [{'a': 0}, {'b': -1}, {'c': 5}, {'a': 0}],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2], object_encoding={'v': 'json',
                                                             'z': 'utf8'})

    pf = ParquetFile(fn)

    # string stats should be stored without byte-encoding
    zcol = [c for c in pf.row_groups[0].columns
            if c.meta_data.path_in_schema == ['z']][0]
    assert zcol.meta_data.statistics.min == b'a'

    result = sorted_partitioned_columns(pf)
    expected = {'x': {'min': [1, 3], 'max': [2, 4]},
                'z': {'min': ['a', 'c'], 'max': ['b', 'd']}}

    # NB column v should not feature, as dict are unorderable
    assert result == expected


def test_sorted_row_group_columns_with_filters(tempdir):
    dd = pytest.importorskip('dask.dataframe')
    # create dummy dataframe
    df = pd.DataFrame({'unique': [0, 0, 1, 1, 2, 2, 3, 3],
                       'id': ['id1', 'id2',
                              'id1', 'id2',
                              'id1', 'id2',
                              'id1', 'id2']},
                      index=[0, 0, 1, 1, 2, 2, 3, 3])
    df = dd.from_pandas(df, npartitions=2)
    fn = os.path.join(tempdir, 'foo.parquet')
    df.to_parquet(fn,
                  engine='fastparquet',
                  partition_on=['id'])
    # load ParquetFile
    pf = ParquetFile(fn)
    filters = [('id', '==', 'id1')]

    # without filters no columns are sorted
    result = sorted_partitioned_columns(pf)
    expected = {}
    assert result == expected

    # with filters both columns are sorted
    result = sorted_partitioned_columns(pf, filters=filters)
    expected = {'__null_dask_index__': {'min': [0, 2], 'max': [1, 3]},
                'unique': {'min': [0, 2], 'max': [1, 3]}}
    assert result == expected


def test_iter(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2], write_index=True)
    pf = ParquetFile(fn)
    out = iter(pf.iter_row_groups(index='index'))
    d1 = next(out)
    pd.testing.assert_frame_equal(d1, df[:2], check_dtype=False, check_index_type=False)
    d2 = next(out)
    pd.testing.assert_frame_equal(d2, df[2:], check_dtype=False, check_index_type=False)
    with pytest.raises(StopIteration):
        next(out)


def test_pickle(tempdir):
    import pickle
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2], write_index=True)
    pf = ParquetFile(fn)
    pf2 = pickle.loads(pickle.dumps(pf))
    assert pf.to_pandas().equals(pf2.to_pandas())


def test_directory_local(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'
    write(os.path.join(tempdir, 'foo1.parquet'), df)
    write(os.path.join(tempdir, 'foo2.parquet'), df)
    pf = ParquetFile(tempdir)
    assert pf.info['rows'] == 8
    assert pf.to_pandas()['z'].tolist() == ['a', 'b', 'c', 'd'] * 2


def test_directory_error(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'
    write(os.path.join(tempdir, 'foo1.parquet'), df)
    write(os.path.join(tempdir, 'foo2.parquet'), df)
    with pytest.raises(ValueError, match="fsspec"):
        ParquetFile(tempdir, open_with=lambda *args: open(*args))


def test_directory_mem():
    import fsspec
    m = fsspec.filesystem("memory")
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'
    write('/dir/foo1.parquet', df, open_with=m.open)
    write('/dir/foo2.parquet', df, open_with=m.open)

    # inferred FS
    pf = ParquetFile("/dir", open_with=m.open)
    assert pf.info['rows'] == 8
    assert pf.to_pandas()['z'].tolist() == ['a', 'b', 'c', 'd'] * 2

    # inferred FS
    pf = ParquetFile("/dir/*", open_with=m.open)
    assert pf.info['rows'] == 8
    assert pf.to_pandas()['z'].tolist() == ['a', 'b', 'c', 'd'] * 2

    # explicit FS
    pf = ParquetFile("/dir", fs=m)
    assert pf.info['rows'] == 8
    assert pf.to_pandas()['z'].tolist() == ['a', 'b', 'c', 'd'] * 2
    m.store.clear()


def test_directory_mem_nest():
    import fsspec
    m = fsspec.filesystem("memory")
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    df.index.name = 'index'
    write('/dir/field=a/foo1.parquet', df, open_with=m.open)
    write('/dir/field=b/foo2.parquet', df, open_with=m.open)

    pf = ParquetFile("/dir", fs=m)
    assert pf.info['rows'] == 8
    assert pf.to_pandas()['z'].tolist() == ['a', 'b', 'c', 'd'] * 2
    assert pf.to_pandas()['field'].tolist() == ['a'] * 4 + ['b'] * 4


def test_attributes(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])
    pf = ParquetFile(fn)
    assert pf.columns == ['x', 'y', 'z']
    assert len(pf.row_groups) == 2
    assert pf.count() == 4
    assert join_path(fn).replace("\\", "/") == pf.info['name']
    assert join_path(fn).replace("\\", "/") in str(pf)
    for col in df:
        assert getattr(pf.dtypes[col], "numpy_dtype", pf.dtypes[col]) == df.dtypes[col]


def test_open_standard(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2], file_scheme='hive',
          open_with=open)
    pf = ParquetFile(fn, open_with=open)
    d2 = pf.to_pandas()
    pd.testing.assert_frame_equal(d2, df, check_dtype=False)


def test_filelike(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3, 4],
                       'y': [1.0, 2.0, 1.0, 2.0],
                       'z': ['a', 'b', 'c', 'd']})
    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2])
    with open(fn, 'rb') as f:
        pf = ParquetFile(f, open_with=open)
        d2 = pf.to_pandas()
        pd.testing.assert_frame_equal(d2, df, check_dtype=False)

    b = io.BytesIO(open(fn, 'rb').read())
    pf = ParquetFile(b, open_with=open)
    d2 = pf.to_pandas()
    pd.testing.assert_frame_equal(d2, df, check_dtype=False)


def test_cast_index(tempdir):
    df = pd.DataFrame({'i8': np.array([1, 2, 3, 4], dtype='uint8'),
                       'i16': np.array([1, 2, 3, 4], dtype='int16'),
                       'i32': np.array([1, 2, 3, 4], dtype='int32'),
                       'i64': np.array([1, 2, 3, 4], dtype='int64'),
                       'f16': np.array([1, 2, 3, 4], dtype='float16'),
                       'f32': np.array([1, 2, 3, 4], dtype='float32'),
                       'f64': np.array([1, 2, 3, 4], dtype='float64'),
                       })
    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df)
    pf = ParquetFile(fn)
    for col in ['i32']: #list(df):
        d = pf.to_pandas(index=col)
        if d.index.dtype.kind == 'i':
            assert d.index.dtype == 'int64'
        elif d.index.dtype.kind == 'u':
            assert d.index.dtype == 'uint64'
        else:
            assert d.index.dtype == 'float64'
        print(col,  (d.index == df[col]).all())

        # assert (d.index == df[col]).all()


def test_zero_child_leaf(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3]})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df)

    pf = ParquetFile(fn)
    assert pf.columns == ['x']

    pf._schema[1].num_children = 0
    assert pf.columns == ['x']


def test_request_nonexistent_column(tempdir):
    df = pd.DataFrame({'x': [1, 2, 3]})

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df)

    pf = ParquetFile(fn)
    with pytest.raises(ValueError):
        pf.to_pandas(columns=['y'])


def test_read_multiple_no_metadata(tempdir):
    df = pd.DataFrame({'x': [1, 5, 2, 5]})
    write(tempdir, df, file_scheme='hive', row_group_offsets=[0, 2])
    os.unlink(os.path.join(tempdir, '_metadata'))
    os.unlink(os.path.join(tempdir, '_common_metadata'))
    import glob
    flist = list(sorted(glob.glob(os.path.join(tempdir, '*'))))
    pf = ParquetFile(flist)
    assert len(pf.row_groups) == 2
    out = pf.to_pandas()
    pd.testing.assert_frame_equal(out, df, check_dtype=False)


def test_single_upper_directory(tempdir):
    df = pd.DataFrame({'x': [1, 5, 2, 5], 'y': ['aa'] * 4})
    write(tempdir, df, file_scheme='hive', partition_on='y')
    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert (out.y == 'aa').all()

    os.unlink(os.path.join(tempdir, '_metadata'))
    os.unlink(os.path.join(tempdir, '_common_metadata'))
    import glob
    flist = list(sorted(glob.glob(os.path.join(tempdir, '*/*'))))
    pf = ParquetFile(flist, root=tempdir)
    assert pf.fn == join_path(os.path.join(tempdir, '_metadata'))
    out = pf.to_pandas()
    assert (out.y == 'aa').all()


def test_numerical_partition_name(tempdir):
    df = pd.DataFrame({'x': [1, 5, 2, 5], 'y1': ['aa', 'aa', 'bb', 'aa']})
    write(tempdir, df, file_scheme='hive', partition_on=['y1'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert out[out.y1 == 'aa'].x.tolist() == [1, 5, 5]
    assert out[out.y1 == 'bb'].x.tolist() == [2]


def test_floating_point_partition_name(tempdir):
    df = pd.DataFrame({'x': [1e99, 5e-10, 2e+2, -0.1], 'y1': ['aa', 'aa', 'bb', 'aa']})
    write(tempdir, df, file_scheme='hive', partition_on=['y1'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert out[out.y1 == 'aa'].x.tolist() == [1e99, 5e-10, -0.1]
    assert out[out.y1 == 'bb'].x.tolist() == [200.0]


@pytest.mark.skipif(WIN, reason="path contains ':'")
def test_datetime_partition_names(tempdir):
    dates = pd.to_datetime(['2015-05-09', '2018-10-15', '2020-10-17', '2015-05-09'])
    df = pd.DataFrame({
        'date': dates,
        'x': [1, 5, 2, 5]
    })
    write(tempdir, df, file_scheme='hive', partition_on=['date'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert set(out.date.tolist()) == set(dates.tolist())
    assert out[out.date == '2015-05-09'].x.tolist() == [1, 5]
    assert out[out.date == '2020-10-17'].x.tolist() == [2]


def test_string_partition_names(tempdir):
    date_strings = ['2015-05-09', '2018-10-15', '2020-10-17', '2015-05-09']
    df = pd.DataFrame({
        'date': date_strings,
        'x': [1, 5, 2, 5]
    })
    write(tempdir, df, file_scheme='hive', partition_on=['date'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert set(out.date.tolist()) == set(date_strings)
    assert out[out.date == '2015-05-09'].x.tolist() == [1, 5]
    assert out[out.date == '2020-10-17'].x.tolist() == [2]


@pytest.mark.parametrize('partitions', [['2017-01-05', '1421'], ['0.7', '10']])
def test_mixed_partition_types(tempdir, partitions):
    df = pd.DataFrame({
        'partitions': partitions,
        'x': [1, 2]
    })
    write(tempdir, df, file_scheme='hive', partition_on=['partitions'])
    out = ParquetFile(tempdir).to_pandas()
    assert (out.sort_values("x").set_index("x").partitions == df.sort_values("x").set_index("x").partitions).all()


def test_filter_without_paths(tempdir):
    fn = os.path.join(tempdir, 'test.parq')
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7],
        'letter': ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    })
    write(fn, df)

    pf = ParquetFile(fn)
    out = pf.to_pandas(filters=[['x', '>', 3]])
    pd.testing.assert_frame_equal(out, df, check_dtype=False)
    out = pf.to_pandas(filters=[['x', '>', 30]])
    assert len(out) == 0


def test_filter_special(tempdir):
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7],
        'symbol': ['NOW', 'OI', 'OI', 'OI', 'NOW', 'NOW', 'OI']
    })
    write(tempdir, df, file_scheme='hive', partition_on=['symbol'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(filters=[('symbol', '==', 'NOW')])
    assert out.x.tolist() == [1, 5, 6]
    assert out.symbol.tolist() == ['NOW', 'NOW', 'NOW']


def test_filter_dates(tempdir):
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7],
        'date': [
            '2015-05-09', '2017-05-15', '2017-05-14',
            '2017-05-13', '2015-05-10', '2015-05-11', '2017-05-12'
        ]
    })
    write(tempdir, df, file_scheme='hive', partition_on=['date'])
    pf = ParquetFile(tempdir)
    out_1 = pf.to_pandas(filters=[('date', '>', '2017-01-01')])

    assert set(out_1.x.tolist()) == {2, 3, 4, 7}
    expected_dates = set(['2017-05-15', '2017-05-14', '2017-05-13', '2017-05-12'])
    assert set(out_1.date.tolist()) == expected_dates

    out_2 = pf.to_pandas(filters=[('date', '==', pd.to_datetime('may 9 2015'))])
    assert out_2.x.tolist() == [1]
    assert out_2.date.tolist() == ['2015-05-09']


def test_in_filter(tempdir):
    symbols = ['a', 'a', 'b', 'c', 'c', 'd']
    values = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame(data={'symbols': symbols, 'values': values})
    write(tempdir, df, file_scheme='hive', partition_on=['symbols'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(filters=[('symbols', 'in', ['a', 'c'])])
    assert set(out.symbols) == {'a', 'c'}


def test_partition_columns(tempdir):
    symbols = ['a', 'a', 'b', 'c', 'c', 'd']
    values = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame(data={'symbols': symbols, 'values': values})
    write(tempdir, df, file_scheme='hive', partition_on=['symbols'])
    pf = ParquetFile(tempdir)

    # partition columns always come after actual columns
    assert pf.to_pandas().columns.tolist() == ['values', 'symbols']
    assert pf.to_pandas(columns=['symbols']).columns.tolist() == ['symbols']
    assert pf.to_pandas(columns=['values']).columns.tolist() == ['values']
    assert pf.to_pandas(columns=[]).columns.tolist() == []


def test_in_filter_numbers(tempdir):
    symbols = ['a', 'a', 'b', 'c', 'c', 'd']
    values = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame(data={'symbols': symbols, 'values': values})
    write(tempdir, df, file_scheme='hive', partition_on=['values'])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(filters=[('values', 'in', ['1', '4'])])
    assert set(out.symbols) == {'a', 'c'}
    out = pf.to_pandas(filters=[('values', 'in', [1, 4])])
    assert set(out.symbols) == {'a', 'c'}


def test_filter_stats(tempdir):
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7],
    })
    write(tempdir, df, file_scheme='hive', row_group_offsets=[0, 4])
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(filters=[('x', '>=', 5)])
    assert out.x.tolist() == [5, 6, 7]


@pytest.mark.parametrize("vals,vmin,vmax,expected_in, expected_not_in", [
    # no stats
    ([3, 6], None, None, False, False),

    # unique values
    ([3, 6], 3, 3, False, True),
    ([3, 6], 2, 2, True, False),

    # open-ended intervals
    ([3, 6], None, 7, False, False),
    ([3, 6], None, 2, True, False),
    ([3, 6], 2, None, False, False),
    ([3, 6], 7, None, True, False),

    # partial matches
    ([3, 6], 2, 4, False, False),
    ([3, 6], 5, 6, False, True),
    ([3, 6], 2, 3, False, True),
    ([3, 6], 6, 7, False, True),

    # non match
    ([3, 6], 1, 2, True, False),
    ([3, 6], 7, 8, True, False),

    # spanning interval
    ([3, 6], 1, 8, False, False),

    # empty values
    ([], 1, 8, True, False),

])
def test_in_filters(vals, vmin, vmax, expected_in, expected_not_in):
    assert filter_in(vals, vmin, vmax) == expected_in
    assert filter_in(list(reversed(vals)), vmin, vmax) == expected_in

    assert filter_not_in(vals, vmin, vmax) == expected_not_in
    assert filter_not_in(list(reversed(vals)), vmin, vmax) == expected_not_in


def test_in_filter_rowgroups(tempdir):
    fn = os.path.join(tempdir, 'test.parq')
    df = pd.DataFrame({
        'x': range(10),
    })
    write(fn, df, row_group_offsets=2)
    pf = ParquetFile(fn)
    row_groups = list(pf.iter_row_groups(filters=[('x', 'in', [2])]))
    assert len(row_groups) == 1
    assert row_groups[0].x.tolist() == [2, 3]

    row_groups = list(pf.iter_row_groups(filters=[('x', 'in', [9])]))
    assert len(row_groups) == 1
    assert row_groups[0].x.tolist() == [8, 9]

    row_groups = list(pf.iter_row_groups(filters=[('x', 'in', [2, 9])]))
    assert len(row_groups) == 2
    assert row_groups[0].x.tolist() == [2, 3]
    assert row_groups[1].x.tolist() == [8, 9]


def test_unexisting_filter_cols(tempdir):
    fn = os.path.join(tempdir, 'test.parq') 
    df = pd.DataFrame({'a': range(5), 'b': [1, 1, 2, 2, 2]})
    write(fn, df, file_scheme='hive', partition_on='b')
    pf = ParquetFile(fn)
    with pytest.raises(ValueError, match="{'c'}.$"):
        rec_df = ParquetFile(fn).to_pandas(filters=[(('a', '>=', 0),
                                                     ('c', '==', 0),)])
    

def test_index_not_in_columns(tempdir):
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': [4, 5, 6]}).set_index('a')
    write(tempdir, df, file_scheme='hive')
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(columns=['b'])
    assert out.index.tolist() == ['x', 'y', 'z']
    out = pf.to_pandas(columns=['b'], index=False)
    assert out.index.tolist() == [0, 1, 2]


def test_no_index_name(tempdir):
    df = pd.DataFrame({'__index_level_0__': ['x', 'y', 'z'],
                       'b': [4, 5, 6]}).set_index('__index_level_0__')
    write(tempdir, df, file_scheme='hive')
    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert out.index.name is None
    assert out.index.tolist() == ['x', 'y', 'z']

    df = pd.DataFrame({'__index_level_0__': ['x', 'y', 'z'],
                       'b': [4, 5, 6]})
    write(tempdir, df, file_scheme='hive')
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(index='__index_level_0__', columns=['b'])
    assert out.index.name is None
    assert out.index.tolist() == ['x', 'y', 'z']

    pf = ParquetFile(tempdir)
    out = pf.to_pandas()
    assert out.index.name is None
    assert out.index.tolist() == [0, 1, 2]


def test_input_column_list_not_mutated(tempdir):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    write(tempdir, df, file_scheme='hive')
    cols = ['a']
    pf = ParquetFile(tempdir)
    out = pf.to_pandas(columns=cols)
    assert cols == ['a']


def test_drill_list(tempdir):
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': [4, 5, 6]})
    dir1 = os.path.join(tempdir, 'x')
    fn1 = os.path.join(dir1, 'part.0.parquet')
    os.makedirs(dir1)
    write(fn1, df)
    dir2 = os.path.join(tempdir, 'y')
    fn2 = os.path.join(dir2, 'part.0.parquet')
    os.makedirs(dir2)
    write(fn2, df)

    pf = ParquetFile([fn1, fn2])
    out = pf.to_pandas()
    assert out.a.tolist() == ['x', 'y', 'z'] * 2
    assert out.dir0.tolist() == ['x'] * 3 + ['y'] * 3


def test_multi_list(tempdir):
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': [4, 5, 6]})
    dir1 = os.path.join(tempdir, 'x')
    write(dir1, df, file_scheme='hive')
    dir2 = os.path.join(tempdir, 'y')
    write(dir2, df, file_scheme='hive')
    dir3 = os.path.join(tempdir, 'z', 'deep')
    write(dir3, df, file_scheme='hive')

    pf = ParquetFile([dir1, dir2])
    out = pf.to_pandas()  # this version may have extra column!
    assert out.a.tolist() == ['x', 'y', 'z'] * 2
    pf = ParquetFile([dir1, dir2, dir3])
    out = pf.to_pandas()
    assert out.a.tolist() == ['x', 'y', 'z'] * 3


def test_hive_and_drill_list(tempdir):
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': [4, 5, 6]})
    dir1 = os.path.join(tempdir, 'x=0')
    fn1 = os.path.join(dir1, 'part.0.parquet')
    os.makedirs(dir1)
    write(fn1, df)
    dir2 = os.path.join(tempdir, 'y')
    fn2 = os.path.join(dir2, 'part.0.parquet')
    os.makedirs(dir2)
    write(fn2, df)

    pf = ParquetFile([fn1, fn2])
    out = pf.to_pandas()
    assert out.a.tolist() == ['x', 'y', 'z'] * 2
    assert out.dir0.tolist() == ['x=0'] * 3 + ['y'] * 3


def test_bad_file_paths(tempdir):
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': [4, 5, 6]})
    dir1 = os.path.join(tempdir, 'x=0')
    fn1 = os.path.join(dir1, 'part.=.parquet')
    os.makedirs(dir1)
    write(fn1, df)
    dir2 = os.path.join(tempdir, 'y/z')
    fn2 = os.path.join(dir2, 'part.0.parquet')
    os.makedirs(dir2)
    write(fn2, df)

    pf = ParquetFile([fn1, fn2])
    assert pf.file_scheme == 'other'
    out = pf.to_pandas()
    assert out.a.tolist() == ['x', 'y', 'z'] * 2
    assert 'dir0' not in out

    path1 = os.path.join(tempdir, 'data')
    fn1 = os.path.join(path1, 'out.parq')
    os.makedirs(path1)
    write(fn1, df)
    path2 = os.path.join(tempdir, 'data2')
    fn2 = os.path.join(path2, 'out.parq')
    os.makedirs(path2)
    write(fn2, df)
    pf = ParquetFile([fn1, fn2])
    out = pf.to_pandas()
    assert out.a.tolist() == ['x', 'y', 'z'] * 2


def test_compression_zstd(tempdir):
    df = pd.DataFrame(
        {
            'x': np.arange(1000),
            'y': np.arange(1, 1001),
            'z': np.arange(2, 1002),
        }
    )

    fn = os.path.join(tempdir, 'foocomp.parquet')

    c = {
        "x": {
            "type": "gzip",
            "args": {
                "compresslevel": 5,
            }
        },
        "y": {
            "type": "zstd",
            "args": {
                "level": 5,
            }
        },
        "_default": {
            "type": "gzip",
            "args": None
        }
    }
    write(fn, df, compression=c)

    p = ParquetFile(fn)

    df2 = p.to_pandas()

    pd.testing.assert_frame_equal(df, df2, check_dtype=False)


def test_compression_lz4(tempdir):
    df = pd.DataFrame(
        {
            'x': np.arange(1000),
            'y': np.arange(1, 1001),
            'z': np.arange(2, 1002),
        }
    )

    fn = os.path.join(tempdir, 'foocomp.parquet')

    c = {
        "x": {
            "type": "gzip",
            "args": {
                "compresslevel": 5,
            }
        },
        "y": {
            "type": "lz4",
            "args": {
                "compression": 5,
                "store_size": False,
            }
        },
        "_default": {
            "type": "gzip",
            "args": None
        }
    }
    write(fn, df, compression=c)

    p = ParquetFile(fn)

    df2 = p.to_pandas()

    pd.testing.assert_frame_equal(df, df2, check_dtype=False)


def test_compression_snappy(tempdir):
    df = pd.DataFrame(
        {
            'x': np.arange(1000),
            'y': np.arange(1, 1001),
            'z': np.arange(2, 1002),
        }
    )

    fn = os.path.join(tempdir, 'foocomp.parquet')

    c = {
        "x": {
            "type": "gzip",
            "args": {
                "compresslevel": 5,
            }
        },
        "y": {
            "type": "snappy",
            "args": None
        },
        "_default": {
            "type": "gzip",
            "args": None
        }
    }
    write(fn, df, compression=c)

    p = ParquetFile(fn)

    df2 = p.to_pandas()

    pd.testing.assert_frame_equal(df, df2, check_dtype=False)


def test_int96_stats(tempdir):
    df = pd.util.testing.makeMixedDataFrame()

    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df, row_group_offsets=[0, 2], times='int96')

    p = ParquetFile(fn)

    s = statistics(p)
    assert isinstance(s['min']['D'][0], (np.datetime64, Timestamp))
    assert 'D' in sorted_partitioned_columns(p)


def test_only_partition_columns(tempdir):
    df = pd.DataFrame({'a': np.random.rand(20),
                       'b': np.random.choice(['hi', 'ho'], size=20),
                       'c': np.random.choice(['a', 'b'], size=20)})
    write(tempdir, df, file_scheme='hive', partition_on=['b'])
    pf = ParquetFile(tempdir)
    df2 = pf.to_pandas(columns=['b'])
    df.b.value_counts().to_dict() == df2.b.value_counts().to_dict()

    write(tempdir, df, file_scheme='hive', partition_on=['a', 'b'])
    pf = ParquetFile(tempdir)
    df2 = pf.to_pandas(columns=['a', 'b'])
    df.b.value_counts().to_dict() == df2.b.value_counts().to_dict()

    df2 = pf.to_pandas(columns=['b'])
    df.b.value_counts().to_dict() == df2.b.value_counts().to_dict()

    df2 = pf.to_pandas(columns=['b', 'c'])
    df.b.value_counts().to_dict() == df2.b.value_counts().to_dict()

    with pytest.raises(ValueError):
        # because this leaves no data to write
        write(tempdir, df[['b']], file_scheme='hive', partition_on=['b'])


def test_path_containing_metadata_df():
    p = ParquetFile(os.path.join(TEST_DATA, "dir_metadata", "empty.parquet"))
    df = p.to_pandas()
    assert list(p.columns) == ['a', 'b', 'c', '__index_level_0__']
    assert len(df) == 0


def test_empty_df():
    p = ParquetFile(os.path.join(TEST_DATA, "empty.parquet"))
    df = p.to_pandas()
    assert list(p.columns) == ['a', 'b', 'c', '__index_level_0__']
    assert len(df) == 0


def test_unicode_cols(tempdir):
    fn = os.path.join(tempdir, 'test.parq')
    df = pd.DataFrame({u"région": [1, 2, 3]})
    write(fn, df)
    pf = ParquetFile(fn)
    pf.to_pandas()


def test_multi_cat(tempdir):
    fn = os.path.join(tempdir, 'test.parq')
    N = 200
    df = pd.DataFrame(
        {'a': np.random.randint(10, size=N),
         'b': np.random.choice(['a', 'b', 'c'], size=N),
         'c': np.arange(200)})
    df['a'] = df.a.astype('category')
    df['b'] = df.b.astype('category')
    df = df.set_index(['a', 'b'])
    write(fn, df)

    pf = ParquetFile(fn)
    df1 = pf.to_pandas()
    assert (df1.index.values == df.index.values).all()
    assert (df1.loc[1, 'a'].values == df.loc[1, 'a'].values).all()


def test_multi_cat_single(tempdir):
    fn = os.path.join(tempdir, 'test.parq')
    N = 200
    df = pd.DataFrame(
        {'a': np.random.randint(10, size=N),
         'b': np.random.choice(['a', 'b', 'c'], size=N),
         'c': np.arange(200)})
    df = df.set_index(['a', 'b'])
    write(fn, df)
    pf = ParquetFile(fn)
    df1 = pf.to_pandas()
    assert (df1.index.values == df.index.values).all()
    assert (df1.loc[1, 'a'].values == df.loc[1, 'a'].values).all()


def test_multi_cat_split(tempdir):
    # like test above, but across multiple row-groups; we test that the
    # categories are consistent
    fn = os.path.join(tempdir, 'test.parq')
    N = 200
    df = pd.DataFrame(
        {'a': np.random.randint(10, size=N),
         'b': np.random.choice(['a', 'b', 'c'], size=N),
         'c': np.arange(200)})
    df = df.set_index(['a', 'b'])
    write(fn, df, row_group_offsets=25)

    pf = ParquetFile(fn)
    df1 = pf.to_pandas()
    assert (df1.index.values == df.index.values).all()
    assert (df1.loc[1, 'a'].values == df.loc[1, 'a'].values).all()


def test_multi(tempdir):
    fn = os.path.join(tempdir, 'test.parq')
    N = 200
    df = pd.DataFrame(
        {'a': np.random.randint(10, size=N),
         'b': np.random.choice(['a', 'b', 'c'], size=N),
         'c': np.arange(200)})
    df = df.set_index(['a', 'b'])
    write(fn, df)

    pf = ParquetFile(fn)
    df1 = pf.to_pandas()
    assert (df1.index.values == df.index.values).all()
    assert (df1.loc[1, 'a'].values == df.loc[1, 'a'].values).all()


def test_simple_nested():
    fn = os.path.join(TEST_DATA, 'nested1.parquet')
    pf = ParquetFile(fn)
    assert len(pf.dtypes) == 5
    out = pf.to_pandas()
    assert len(out.columns) == 5
    assert '_adobe_corpnew' not in out.columns
    assert all('_adobe_corpnew' + '.' in c for c in out.columns)


def test_pandas_metadata_inference():
    fn = os.path.join(TEST_DATA, 'metas.parq')
    df = ParquetFile(fn).to_pandas()
    assert df.columns.name == 'colindex'
    assert df.index.name == 'rowindex'
    assert df.index.tolist() == [2, 3]

    df = ParquetFile(fn).to_pandas(index='a')
    assert df.index.name == 'a'
    assert df.columns.name == 'colindex'

    df = ParquetFile(fn).to_pandas(index=False)
    assert df.index.tolist() == [0, 1]
    assert df.index.name is None


def test_write_index_false(tempdir):
    fn = os.path.join(tempdir, 'test.parquet')
    df = pd.DataFrame(0, columns=['a'], index=range(1, 3))
    write(fn, df, write_index=False)
    rec_df = ParquetFile(fn).to_pandas()
    assert rec_df.index[0] == 0


def test_timestamp_filer(tempdir):
    fn = os.path.join(tempdir, 'test.parquet')
    ts = [pd.Timestamp('2021/01/01 08:00:00'),
          pd.Timestamp('2021/01/05 10:00:00')]
    val = [10, 34]
    df = pd.DataFrame({'val': val, 'ts': ts})
    # two row-groups
    write(fn, df, row_group_offsets=1, file_scheme='hive')

    ts_filter = pd.Timestamp('2021/01/03 00:00:00')
    pf = ParquetFile(fn)
    filt = [[('ts', '<', ts_filter)], [('ts', '>=', ts_filter)]]
    assert pf.to_pandas(filters=filt).val.tolist() == [10, 34]

    filt = [[('ts', '>=', ts_filter)], [('ts', '<', ts_filter)]]
    assert pf.to_pandas(filters=filt).val.tolist() == [10, 34]

    ts_filter_down = pd.Timestamp('2021/01/03 00:00:00')
    ts_filter_up = pd.Timestamp('2021/01/06 00:00:00')
    # AND filter
    filt = [[('ts', '>=', ts_filter_down), ('ts', '<', ts_filter_up)]]
    assert pf.to_pandas(filters=filt).val.tolist() == [34]


@pytest.mark.xfail(condition=fastparquet.writer.DATAPAGE_VERSION == 2, reason="not implemented")
def test_row_filter(tempdir):
    fn = os.path.join(tempdir, 'test.parquet')
    df = pd.DataFrame({
        'a': ['o'] * 10 + ['i'] * 5,
        'b': range(15)
    })
    write(fn, df, row_group_offsets=8)
    pf = ParquetFile(fn)
    assert pf.count(filters=[["a", "==", "o"]]) == 15
    assert pf.count(filters=[["a", "==", "o"]], row_filter=True) == 10
    assert pf.count(filters=[["a", "==", "i"]], row_filter=True) == 5
    assert pf.count(filters=[["b", "in", [1, 3, 4]]]) == 8
    assert pf.count(filters=[["b", "in", [1, 3, 4]]], row_filter=True) == 3
    assert pf.to_pandas(filters=[["b", "in", [1, 3, 4]]], row_filter=True
                        ).b.tolist() == [1, 3, 4]
    assert pf.to_pandas(filters=[["a", "<", "o"]], row_filter=True).b.tolist() == [
        10, 11, 12, 13, 14
    ]


def test_select(tempdir):
    fn = os.path.join(tempdir, 'test.parquet')
    val = [2, 10, 34, 76]
    df = pd.DataFrame({'val': val})
    write(fn, df, row_group_offsets=1)

    pf = ParquetFile(fn)
    assert len(pf[0].row_groups) == 1
    assert pf[0].to_pandas().val.tolist() == [2]
    assert pf[1].to_pandas().val.tolist() == [10]
    assert pf[-1].to_pandas().val.tolist() == [76]
    assert pf[:].to_pandas().val.tolist() == val
    assert pf[::2].to_pandas().val.tolist() == val[::2]


def test_head(tempdir):
    fn = os.path.join(tempdir, 'test.parquet')
    val = [2, 10, 34, 76]
    df = pd.DataFrame({'val': val})
    write(fn, df)

    pf = ParquetFile(fn)
    assert pf.head(1).val.tolist() == [2]


def test_spark_date_empty_rg():
    # https://github.com/dask/fastparquet/issues/634
    # first file has header size much smaller than others as it contains no row groups
    fn = os.path.join(TEST_DATA, 'spark-date-empty-rg.parq')
    pf = ParquetFile(fn)
    out = pf.to_pandas(columns=['Date'])
    assert out.Date.tolist() == [pd.Timestamp("2020-1-1"), pd.Timestamp("2020-1-2")]

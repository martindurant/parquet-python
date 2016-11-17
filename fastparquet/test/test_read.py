"""test_read_support.py - unit and integration tests for reading parquet data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
from itertools import product
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
import tempfile
import unittest

import pandas as pd
import pytest

import fastparquet

TEST_DATA = "test-data"


@pytest.yield_fixture()
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


def test_header_magic_bytes(tempdir):
    """Test reading the header magic bytes."""
    fn = os.path.join(tempdir, 'temp.parq')
    with open(fn, 'wb') as f:
        f.write(b"PAR1_some_bogus_data")
    with pytest.raises(fastparquet.ParquetException):
        p = fastparquet.ParquetFile(fn, verify=True)


def test_read_footer():
    """Test reading the footer."""
    p = fastparquet.ParquetFile(os.path.join(TEST_DATA, "nation.impala.parquet"))
    snames = {"schema", "n_regionkey", "n_name", "n_nationkey", "n_comment"}
    assert {s.name for s in p.schema} == snames
    assert set(p.columns) == snames - {"schema"}

files = [os.path.join(TEST_DATA, p) for p in
         ["gzip-nation.impala.parquet", "nation.dict.parquet",
          "nation.impala.parquet", "nation.plain.parquet",
          "snappy-nation.impala.parquet"]]
csvfile = os.path.join(TEST_DATA, "nation.csv")
cols = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]
expected = pd.read_csv(csvfile, delimiter="|", index_col=0, names=cols)


@pytest.yield_fixture()
def s3():
    s3fs = pytest.importorskip('s3fs')
    moto = pytest.importorskip('moto')
    m = moto.mock_s3()
    m.start()
    s3 = s3fs.S3FileSystem()
    s3.mkdir(TEST_DATA)
    paths = []
    for cat, catnum in product(('fred', 'freda'), ('1', '2', '3')):
        path = os.sep.join([TEST_DATA, 'split', 'cat=' + cat,
                            'catnum=' + catnum])
        files = os.listdir(path)
        for fn in files:
            full_path = os.path.join(path, fn)
            s3.put(full_path, full_path)
            paths.append(full_path)
    path = os.path.join(TEST_DATA, 'split')
    files = os.listdir(path)
    for fn in files:
        full_path = os.path.join(path, fn)
        if os.path.isdir(full_path):
            continue
        s3.put(full_path, full_path)
        paths.append(full_path)
    yield s3
    for path in paths:
        s3.rm(path)



def test_read_s3(s3):
    s3fs = pytest.importorskip('s3fs')
    s3 = s3fs.S3FileSystem()
    myopen = s3.open
    pf = fastparquet.ParquetFile(TEST_DATA+'/split/_metadata', open_with=myopen)
    df = pf.to_pandas()
    assert df.shape == (2000, 3)
    assert (df.cat.value_counts() == [1000, 1000]).all()


@pytest.mark.parametrize("parquet_file", files)
def test_file_csv(parquet_file):
    """Test the various file times
    """
    p = fastparquet.ParquetFile(parquet_file)
    data = p.to_pandas()
    if 'comment_col' in data.columns:
        mapping = {'comment_col': "n_comment", 'name': 'n_name',
                   'nation_key': 'n_nationkey', 'region_key': 'n_regionkey'}
        data.columns = [mapping[k] for k in data.columns]
    data.set_index('n_nationkey', inplace=True)

    # FIXME: in future, reader will return UTF8 strings
    for col in cols[1:]:
        if isinstance(data[col][0], bytes):
            data[col] = data[col].str.decode('utf8')
        assert (data[col] == expected[col]).all()


def test_null_int():
    """Test reading a file that contains null records."""
    p = fastparquet.ParquetFile(os.path.join(TEST_DATA, "test-null.parquet"))
    data = p.to_pandas()
    expected = pd.DataFrame([{"foo": 1, "bar": 2}, {"foo": 1, "bar": None}])
    for col in data:
        assert (data[col] == expected[col])[~expected[col].isnull()].all()
        assert sum(data[col].isnull()) == sum(expected[col].isnull())


def test_converted_type_null():
    """Test reading a file that contains null records for a plain column that
     is converted to utf-8."""
    p = fastparquet.ParquetFile(os.path.join(TEST_DATA,
                                         "test-converted-type-null.parquet"))
    data = p.to_pandas()
    expected = pd.DataFrame([{"foo": "bar"}, {"foo": None}])
    for col in data:
        if isinstance(data[col][0], bytes):
            # Remove when re-implemented converted types
            data[col] = data[col].str.decode('utf8')
        assert (data[col] == expected[col])[~expected[col].isnull()].all()
        assert sum(data[col].isnull()) == sum(expected[col].isnull())


def test_null_plain_dictionary():
    """Test reading a file that contains null records for a plain dictionary
     column."""
    p = fastparquet.ParquetFile(os.path.join(TEST_DATA,
                                         "test-null-dictionary.parquet"))
    data = p.to_pandas()
    expected = pd.DataFrame([{"foo": None}] + [{"foo": "bar"},
                             {"foo": "baz"}] * 3)
    for col in data:
        if isinstance(data[col][1], bytes):
            # Remove when re-implemented converted types
            data[col] = data[col].str.decode('utf8')
        assert (data[col] == expected[col])[~expected[col].isnull()].all()
        assert sum(data[col].isnull()) == sum(expected[col].isnull())


def test_dir_partition():
    """Test creation of categories from directory structure"""
    x = np.arange(2000)
    df = pd.DataFrame({
        'num': x,
        'cat': pd.Series(np.array(['fred', 'freda'])[x%2], dtype='category'),
        'catnum': pd.Series(np.array([1, 2, 3])[x%3], dtype='category')})
    pf = fastparquet.ParquetFile(os.path.join(TEST_DATA, "split"))
    out = pf.to_pandas()
    for cat, catnum in product(['fred', 'freda'], [1, 2, 3]):
        assert (df.num[(df.cat==cat) & (df.catnum==catnum)].tolist()) ==\
               out.num[(out.cat==cat) & (out.catnum==catnum)].tolist()
    assert out.cat.dtype == 'category'
    assert out.catnum.dtype == 'category'
    assert out.catnum.cat.categories.dtype == 'int64'


def test_stat_filters():
    path = os.path.join(TEST_DATA, 'split')
    pf = fastparquet.ParquetFile(path)
    base_shape = len(pf.to_pandas())

    filters = [('num', '>', 0)]
    assert len(pf.to_pandas(filters=filters)) == base_shape

    filters = [('num', '<', 0)]
    assert len(pf.to_pandas(filters=filters)) == 0

    filters = [('num', '>', 500)]
    assert 0 < len(pf.to_pandas(filters=filters)) < base_shape

    filters = [('num', '>', 1500)]
    assert 0 < len(pf.to_pandas(filters=filters)) < base_shape

    filters = [('num', '>', 2000)]
    assert len(pf.to_pandas(filters=filters)) == 0

    filters = [('num', '>=', 1999)]
    assert 0 < len(pf.to_pandas(filters=filters)) < base_shape

    filters = [('num', '!=', 1000)]
    assert len(pf.to_pandas(filters=filters)) == base_shape

    filters = [('num', 'in', [-1, -2])]
    assert len(pf.to_pandas(filters=filters)) == 0

    filters = [('num', 'not in', [-1, -2])]
    assert len(pf.to_pandas(filters=filters)) == base_shape

    filters = [('num', 'in', [0])]
    l = len(pf.to_pandas(filters=filters))
    assert 0 < l < base_shape

    filters = [('num', 'in', [0, 1500])]
    assert l < len(pf.to_pandas(filters=filters)) < base_shape

    filters = [('num', 'in', [-1, 2000])]
    assert len(pf.to_pandas(filters=filters)) == base_shape


def test_cat_filters():
    path = os.path.join(TEST_DATA, 'split')
    pf = fastparquet.ParquetFile(path)
    base_shape = len(pf.to_pandas())

    filters = [('cat', '==', 'freda')]
    assert len(pf.to_pandas(filters=filters)) == 1000

    filters = [('cat', '!=', 'freda')]
    assert len(pf.to_pandas(filters=filters)) == 1000

    filters = [('cat', 'in', ['fred', 'freda'])]
    assert 0 < len(pf.to_pandas(filters=filters)) == 2000

    filters = [('cat', 'not in', ['fred', 'frederick'])]
    assert 0 < len(pf.to_pandas(filters=filters)) == 1000

    filters = [('catnum', '==', 2000)]
    assert len(pf.to_pandas(filters=filters)) == 0

    filters = [('catnum', '>=', 2)]
    assert 0 < len(pf.to_pandas(filters=filters)) == 1333

    filters = [('catnum', '>=', 1)]
    assert len(pf.to_pandas(filters=filters)) == base_shape

    filters = [('catnum', 'in', [0, 1])]
    assert len(pf.to_pandas(filters=filters)) == 667

    filters = [('catnum', 'not in', [1, 2, 3])]
    assert len(pf.to_pandas(filters=filters)) == 0

    filters = [('cat', '==', 'freda'), ('catnum', '>=', 2.5)]
    assert len(pf.to_pandas(filters=filters)) == 333

    filters = [('cat', '==', 'freda'), ('catnum', '!=', 2.5)]
    assert len(pf.to_pandas(filters=filters)) == 1000


def test_statistics(tempdir):
    s = pd.Series([b'a', b'b', b'c']*20)
    df = pd.DataFrame({'a': s, 'b': s.astype('category'),
                       'c': s.astype('category').cat.as_ordered()})
    fastparquet.write(tempdir, df, file_scheme='hive')
    pf = fastparquet.ParquetFile(tempdir)
    stat = pf.statistics
    assert stat['max']['a'] == [b'c']
    assert stat['min']['a'] == [b'a']
    assert stat['max']['b'] == [None]
    assert stat['min']['b'] == [None]
    assert stat['max']['c'] == [b'c']
    assert stat['min']['c'] == [b'a']


def test_grab_cats(tempdir):
    s = pd.Series(['a', 'c', 'b']*20)
    df = pd.DataFrame({'a': s, 'b': s.astype('category'),
                       'c': s.astype('category').cat.as_ordered()})
    fastparquet.write(tempdir, df, file_scheme='hive')
    pf = fastparquet.ParquetFile(tempdir)
    cats = pf.grab_cats(['b', 'c'])
    assert (cats['b'] == df.b.cat.categories).all()
    assert (cats['c'] == df.c.cat.categories).all()

"""testing parquet to/from pyspark"""
import datetime
import numpy as np
import os
import pandas as pd
import pytest

import fastparquet
from fastparquet import write
from fastparquet.compression import compressions
from fastparquet.test.util import sql, s3, tempdir, TEST_DATA


def test_map_array(sql):
    """
from pyspark.sql.types import *
df_schema = StructType([
    StructField('map_op_op', MapType(StringType(), StringType(), True), True),
    StructField('map_op_req', MapType(StringType(), StringType(), False), True),
    StructField('map_req_op', MapType(StringType(), StringType(), True), False),
    StructField('map_req_req', MapType(StringType(), StringType(), False), False),
    StructField('arr_op_op', ArrayType(StringType(), True), True),
    StructField('arr_op_req', ArrayType(StringType(), False), True),
    StructField('arr_req_op', ArrayType(StringType(), True), False),
    StructField('arr_req_req', ArrayType(StringType(), False), False)])
keys = ['fred', 'wilma', 'barney', 'betty']
vals = ['franky', 'benji', 'mighty']
out = []
for i in range(1000):
    part = []
    for field in [f.name for f in df_schema.fields]:
        sort, nullable, nullvalue = field.split('_')
        if nullable == 'op' and np.random.random() < 0.3:
            part.append(None)
            continue
        N = np.random.randint(5)
        ks = np.random.choice(keys, size=N).tolist()
        vs = np.random.choice(vals + [None] if nullvalue == 'op' else vals,
                              size=N).tolist()
        if sort == 'map':
            part.append({k: v for (k, v) in zip(ks, vs)})
        else:
            part.append(vs)
    out.append(part)
df = sql.createDataFrame(out, df_schema)
    """
    fn = os.path.join(TEST_DATA, 'map_array.parq')
    expected = sql.read.parquet(fn).toPandas()
    pf = fastparquet.ParquetFile(fn)
    data = pf.to_pandas()
    pd.testing.assert_frame_equal(data, expected)


def test_nested_list(sql):
    """
j = {'nest': {'thing': ['hi', 'world']}}
open('temp.json', 'w').write('\n'.join([json.dumps(j)] * 10))
df = sql.read.json('temp.json')
    """
    fn = os.path.join(TEST_DATA, 'nested.parq')
    pf = fastparquet.ParquetFile(fn)
    assert pf.columns == ['nest.thing']  # NOT contain 'nest'
    out = pf.to_pandas(columns=['nest.thing'])
    assert all([o == ['hi', 'world'] for o in out['nest.thing']])


comps = ['UNCOMPRESSED', 'GZIP', 'SNAPPY']


@pytest.mark.parametrize('scheme', ['simple', 'hive'])
@pytest.mark.parametrize('row_groups', [[0], [0, 500]])
@pytest.mark.parametrize('comp', comps)
@pytest.mark.skipif(os.name == 'nt', reason="don't spark on windows")
def test_writer_to_spark(tempdir, scheme, row_groups, comp, sql):
    data = pd.DataFrame({'i32': np.random.randint(-2**17, 2**17, size=1001,
                                                  dtype=np.int32),
                         'i64': np.random.randint(-2**33, 2**33, size=1001,
                                                  dtype=np.int64),
                         'f': np.random.randn(1001),
                         'bhello': np.random.choice([b'hello', b'you',
                            b'people'], size=1001).astype("O"),
                         't': [datetime.datetime.now()]*1001})

    data['t'] += pd.to_timedelta('1ns')
    data['hello'] = data.bhello.str.decode('utf8')
    data.loc[100, 'f'] = np.nan
    if fastparquet.writer.DATAPAGE_VERSION == 1:
        # unknown failure for dict columns; roundtrips OK with pyarrow
        data['bcat'] = data.bhello.astype('category')
        data['cat'] = data.hello.astype('category')

    fname = os.path.join(tempdir, 'test.parquet')
    write(fname, data, file_scheme=scheme, row_group_offsets=row_groups,
          compression=comp, times='int96', write_index=True)

    df = sql.read.parquet(fname)
    ddf = df.sort('index').toPandas()
    for col in data:
        if data[col].dtype.kind == "M":
            # pyspark auto-converts timezones
            offset = round((datetime.datetime.utcnow() -
                            datetime.datetime.now()).seconds / 3600)
            ddf[col] + datetime.timedelta(hours=offset) == data[col]
        else:
            assert (ddf[col] == data[col])[~ddf[col].isnull()].all()


@pytest.mark.parametrize("int96", ["true", "false"])
@pytest.mark.parametrize("legacy", ["true", "false"])
@pytest.mark.parametrize("version", ["v1"])  # "v2" doesn't do anything!
@pytest.mark.skipif(os.name == 'nt', reason="don't spark on windows")
def test_read_from_spark(tempdir, sql, int96, legacy, version):
    sql.setConf("spark.sql.parquet.int96AsTimestamp", int96)
    sql.setConf("spark.sql.parquet.writeLegacyFormat", legacy)
    sql.setConf("spark.hadoop.parquet.writer.version", version)
    sql.setConf('spark.hadoop.parquet.enable.summary-metadata', 'true')
    data = pd.DataFrame({
        'i32': np.random.randint(-2**17, 2**17, size=1001,
                                 dtype=np.int32),
        'i64': np.random.randint(-2**33, 2**33, size=1001,
                                 dtype=np.int64),
        'f': np.random.randn(1001),
        'bhello': np.random.choice([b'hello', b'you',
                                    b'people'], size=1001).astype("O"),
        }
    )

    data['hello'] = data.bhello.str.decode('utf8')
    data.loc[100, 'f'] = np.nan
    fname = os.path.join(tempdir, 'test.parquet')
    df = sql.createDataFrame(data).repartition(1)
    df.write.parquet(fname)
    files = [f for f in os.listdir(fname) if f.endswith("parquet")]
    fname = os.path.join(fname, files[0])
    ddf = fastparquet.ParquetFile(fname).to_pandas()
    for col in data:
        assert (ddf[col] == data[col])[~ddf[col].isnull()].all()


@pytest.mark.skipif(os.name == 'nt', reason="don't spark on windows")
def test_empty_row_groups(tempdir, sql):
    fn = os.path.join(tempdir, 'output.parquet')
    d0 = pd.DataFrame({'name': ['alice'], 'age': [20]})
    df = sql.createDataFrame(d0)
    df.write.parquet(fn)
    import glob
    files = glob.glob(os.path.join(fn, '*.parquet'))
    sizes = [os.stat(p).st_size for p in files]
    msize = max(sizes)
    pf = fastparquet.ParquetFile(files)  # don't necessarily have metadata
    assert len(files) > 1  # more than one worker was writing
    d = pf.to_pandas(index=False)
    pd.testing.assert_frame_equal(d, d0)

    # destroy empty files
    [os.unlink(f) for (f, s) in zip(files, sizes) if s < msize]

    # loads anyway, since empty row-groups are not touched
    d = pf.to_pandas()
    pd.testing.assert_frame_equal(d, d0)

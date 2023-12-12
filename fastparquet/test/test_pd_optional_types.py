import os
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.core.arrays import IntegerArray
import fastparquet as fp
from .util import tempdir
from fastparquet import write, parquet_thrift
from fastparquet.parquet_thrift.parquet import ttypes as tt
import numpy.random as random


EXPECTED_SERIES_INT8 = random.uniform(low=-128, high=127, size=100).round()
EXPECTED_SERIES_INT16 = random.uniform(low=-32768, high=32767, size=100).round()
EXPECTED_SERIES_INT32 = random.uniform(low=-2147483648, high=2147483647, size=100).round()
EXPECTED_SERIES_INT64 = random.uniform(low=-9223372036854775808, high=9223372036854775807, size=100).round()
EXPECTED_SERIES_UINT8 = random.uniform(low=0, high=255, size=100).round()
EXPECTED_SERIES_UINT16 = random.uniform(low=0, high=65535, size=100).round()
EXPECTED_SERIES_UINT32 = random.uniform(low=0, high=4294967295, size=100).round()
EXPECTED_SERIES_UINT64 = random.uniform(low=0, high=18446744073709551615, size=100).round()
EXPECTED_SERIES_BOOL = random.choice([False, True], 100)
EXPECTED_SERIES_STRING = random.choice([
    'You', 'are', 'my', 'fire', 
    'The', 'one', 'desire', 
    'Believe', 'when', 'I', 'say', 
    'I', 'want', 'it', 'that', 'way'
    ], 100)


EXPECTED_SERIES_INT8[20:30] = np.nan
EXPECTED_SERIES_INT16[20:30] = np.nan
EXPECTED_SERIES_INT32[20:30] = np.nan
EXPECTED_SERIES_INT64[20:30] = np.nan
EXPECTED_SERIES_UINT8[20:30] = np.nan
EXPECTED_SERIES_UINT16[20:30] = np.nan
EXPECTED_SERIES_UINT32[20:30] = np.nan
EXPECTED_SERIES_UINT64[20:30] = np.nan
EXPECTED_SERIES_BOOL[20:30] = np.nan
EXPECTED_SERIES_STRING[20:30] = np.nan
mask = EXPECTED_SERIES_UINT64 > -1


TEST = pd.DataFrame({
    'int8': pd.Series(pd.array(EXPECTED_SERIES_INT8, dtype='Int8')),
    'int16': pd.Series(pd.array(EXPECTED_SERIES_INT16, dtype='Int16')),
    'int32': pd.Series(pd.array(EXPECTED_SERIES_INT32, dtype='Int32')),
    'int64': pd.Series(pd.array(EXPECTED_SERIES_INT64, dtype='Int64')),
    'uint8': pd.Series(pd.array(EXPECTED_SERIES_UINT8, dtype='UInt8')),
    'uint16': pd.Series(pd.array(EXPECTED_SERIES_UINT16, dtype='UInt16')),
    'uint32': pd.Series(pd.array(EXPECTED_SERIES_UINT32, dtype='UInt32')),
    'uint64': pd.Series(pd.array(EXPECTED_SERIES_UINT64, dtype='UInt64')),
    'bool': pd.Series(pd.array(EXPECTED_SERIES_BOOL, dtype='boolean')),
    'string': pd.Series(EXPECTED_SERIES_STRING, dtype='string')
})


EXPECTED = TEST


EXPECTED_PARQUET_TYPES = {
    'int8': 'INT32',
    'int16': 'INT32',
    'int32': 'INT32',
    'int64': 'INT64',
    'uint8': 'INT32',
    'uint16': 'INT32',
    'uint32': 'INT32',
    'uint64': 'INT64',
    'bool': 'BOOLEAN',
    'string': 'BYTE_ARRAY'
}


@pytest.mark.parametrize('comp', (None, 'snappy', 'gzip'))
@pytest.mark.parametrize('scheme', ('simple', 'hive'))
def test_write_nullable_columns(tempdir, scheme, comp):
    fname = os.path.join(tempdir, 'test_write_nullable_columns.parquet')
    write(fname, TEST, file_scheme=scheme, compression=comp)
    pf = fp.ParquetFile(fname)
    df = pf.to_pandas()
    pq_types = {
        se.name: tt.Type._VALUES_TO_NAMES[se.type]
        for se in pf.schema.schema_elements
        if se.type is not None
    }
    assert_frame_equal(EXPECTED, df, check_index_type=False, check_dtype=False)
    assert pq_types == EXPECTED_PARQUET_TYPES

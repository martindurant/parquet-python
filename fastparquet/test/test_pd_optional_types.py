import os
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import fastparquet as fp
from .util import tempdir
from fastparquet import write, parquet_thrift
from fastparquet.parquet_thrift.parquet import ttypes as tt
import numpy.random as random


EXPECTED_SERIES_INT8 = pd.Series(random.uniform(low=-128, high=127,size=100)).round()
EXPECTED_SERIES_INT16 = pd.Series(random.uniform(low=-32768, high=32767,size=100)).round()
EXPECTED_SERIES_INT32 = pd.Series(random.uniform(low=-2147483648, high=2147483647,size=100)).round()
EXPECTED_SERIES_INT64 = pd.Series(random.uniform(low=-9223372036854775808, high=9223372036854775807,size=100)).round()
EXPECTED_SERIES_UINT8 = pd.Series(random.uniform(low=0, high=255,size=100)).round()
EXPECTED_SERIES_UINT16 = pd.Series(random.uniform(low=0, high=65535,size=100)).round()
EXPECTED_SERIES_UINT32 = pd.Series(random.uniform(low=0, high=4294967295,size=100)).round()
EXPECTED_SERIES_UINT64 = pd.Series(random.uniform(low=0, high=18446744073709551615,size=100)).round()
EXPECTED_SERIES_BOOL = pd.Series(random.choice([False, True], 100))
EXPECTED_SERIES_STRING = pd.Series(random.choice([
    'You', 'are', 'my', 'fire', 
    'The', 'one', 'desire', 
    'Believe', 'when', 'I', 'say', 
    'I', 'want', 'it', 'that', 'way'
    ], 100))


EXPECTED_SERIES_INT8.loc[20:30] = np.nan
EXPECTED_SERIES_INT16.loc[20:30] = np.nan
EXPECTED_SERIES_INT32.loc[20:30] = np.nan
EXPECTED_SERIES_INT64.loc[20:30] = np.nan
EXPECTED_SERIES_UINT8.loc[20:30] = np.nan
EXPECTED_SERIES_UINT16.loc[20:30] = np.nan
EXPECTED_SERIES_UINT32.loc[20:30] = np.nan
EXPECTED_SERIES_UINT64.loc[20:30] = np.nan
EXPECTED_SERIES_BOOL.loc[20:30] = np.nan
EXPECTED_SERIES_STRING.loc[20:30] = np.nan


TEST = pd.DataFrame({
    'int8': EXPECTED_SERIES_INT8.astype('Int8'),
    'int16': EXPECTED_SERIES_INT16.astype('Int16'),
    'int32': EXPECTED_SERIES_INT32.astype('Int32'),
    'int64': EXPECTED_SERIES_INT64.astype('Int64'),
    'uint8': EXPECTED_SERIES_UINT8.astype('UInt8'),
    'uint16': EXPECTED_SERIES_UINT16.astype('UInt16'),
    'uint32': EXPECTED_SERIES_UINT32.astype('UInt32'),
    'uint64': EXPECTED_SERIES_UINT64.astype('UInt64'),
    'bool': EXPECTED_SERIES_BOOL.astype('boolean'),
    'string': EXPECTED_SERIES_STRING.astype('string')
})


EXPECTED = pd.DataFrame({
    'int8': EXPECTED_SERIES_INT8.astype('float16'),
    'int16': EXPECTED_SERIES_INT16.astype('float32'),
    'int32': EXPECTED_SERIES_INT32.astype('float64'),
    'int64': EXPECTED_SERIES_INT64.astype('float64'),
    'uint8': EXPECTED_SERIES_UINT8.astype('float16'),
    'uint16': EXPECTED_SERIES_UINT16.astype('float32'),
    'uint32': EXPECTED_SERIES_UINT32.astype('float64'),
    'uint64': EXPECTED_SERIES_UINT64.astype('float64'),
    'bool': EXPECTED_SERIES_BOOL.astype('float16'),
    'string': EXPECTED_SERIES_STRING
})


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

@pytest.mark.parametrize('comp', (None,'snappy', 'gzip'))
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

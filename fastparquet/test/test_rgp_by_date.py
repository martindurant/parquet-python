"""
   test_rgp_by_date.py
   Tests for row grouping by date in parquet data writing.
"""

import os
import pandas as pd

from fastparquet import write, ParquetFile
from fastparquet.util import previous_date_offset
from fastparquet.test.util import tempdir

def test_midnight_anchoring():
    """Test correct anchoring for miscellaneous date offsets."""
    ts = pd.Timestamp('2020/01/02 08:59')
    on_offset = pd.Timestamp('2020/01/02 08:40')    # test 20 minutes offset
    assert previous_date_offset(ts, '20T', True, True) == on_offset
    on_offset = pd.Timestamp('2020/01/02 08:00')    # test 2 hours offset
    assert previous_date_offset(ts, '2H', True, True) == on_offset
    on_offset = pd.Timestamp('2020/01/02 00:00')    # test 1 day offset
    assert previous_date_offset(ts, '1D', True, True) == on_offset
    on_offset = pd.Timestamp('2020/01/01 00:00')    # test 1 month offset
    assert previous_date_offset(ts, '1M', True, True) == on_offset

def test_write_with_rgp_by_date_as_index(tempdir):
    fn = os.path.join(tempdir, 'foo.parquet')
    offset = '2H'

    # Case 1 - Date data used for grouping is DataFrame index.    
    # Step 1 - Writing of a 1st df and testing correct grouping.
    # df1 is not sorted: it will be sorted before writing.
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9],
                        'pressure': [1e5, 1.1e5, 0.95e5],
                        'location': ['Paris', 'Paris', 'Milan']},
                        index = [pd.Timestamp('2020/01/02 01:59:00'),
                                 pd.Timestamp('2020/01/02 03:59:00'),
                                 pd.Timestamp('2020/01/02 02:59:00')])
    write(fn, df1, row_group_offsets=offset, file_scheme='hive')
    pf = ParquetFile(fn)
    expected_1_1 = pd.DataFrame({'index': [pd.Timestamp('2020/01/02 01:59:00')],
                                 'humidity': [0.3],
                                 'pressure': [1e5],
                                 'location': ['Paris']})
    expected_1_1.index.name = 'index'
    recorded_1_1 = pf.read_row_group_file(pf.row_groups[0], pf.columns, pf.categories)
    assert expected_1_1.equals(recorded_1_1)
    expected_1_2 = pd.DataFrame({'index': [pd.Timestamp('2020/01/02 02:59:00'),
                                           pd.Timestamp('2020/01/02 03:59:00')],
                                 'humidity': [0.9, 0.8],
                                 'pressure': [0.95e5, 1.1e5],
                                 'location': ['Milan', 'Paris']})
    expected_1_2.index.name = 'index'
    recorded_1_2 = pf.read_row_group_file(pf.row_groups[1], pf.columns, pf.categories)
    assert expected_1_2.equals(recorded_1_2)

    # Step 2 - Updating with a 2nd df having overlapping data and duplicates
    #          and testing correct row group insertion and removal of duplicates.
    # df2 is not sorted: it will be sorted before writing.
    df2 = pd.DataFrame({'humidity': [0.5, 0.3, 0.4, 0.8, 1.1],
                        'pressure': [9e4, 1e5, 1.1e5, 1.1e5, 0.95e5],
                        'location': ['Tokyo', 'Paris', 'Paris', 'Paris', 'Paris']},
                        index = [pd.Timestamp('2020/01/01 01:59:00'),
                                 pd.Timestamp('2020/01/02 01:58:00'),
                                 pd.Timestamp('2020/01/02 01:59:00'),
                                 pd.Timestamp('2020/01/02 00:38:00'),
                                 pd.Timestamp('2020/01/03 02:59:00')])
    write(fn, df2, row_group_offsets=offset, file_scheme='hive', append=True,
          drop_duplicates_on = 'index')
    pf = ParquetFile(fn)
    expected_2_1 = pd.DataFrame({'index': [pd.Timestamp('2020/01/01 01:59:00')],
                                 'humidity': [0.5],
                                 'pressure': [9e4],
                                 'location': ['Tokyo']})
    expected_2_1.index.name = 'index'
    recorded_2_1 = pf.read_row_group_file(pf.row_groups[0], pf.columns, pf.categories)
    assert expected_2_1.equals(recorded_2_1) # test row group insert in 1st position
    expected_2_2 = pd.DataFrame({'index': [pd.Timestamp('2020-01-02 00:38:00'),
                                           pd.Timestamp('2020-01-02 01:58:00'),
                                           pd.Timestamp('2020-01-02 01:59:00')],
                                 'humidity': [0.8, 0.3, 0.4],
                                 'pressure': [1.1e5, 1e5, 1.1e5],
                                 'location': ['Paris','Paris','Paris']})
    expected_2_2.index.name = 'index'
    recorded_2_2 = pf.read_row_group_file(pf.row_groups[1], pf.columns, pf.categories)
    assert expected_2_2.equals(recorded_2_2) # test drop duplicate according timestamp and keep last.

def test_write_with_rgp_by_date_as_col(tempdir):
    fn = os.path.join(tempdir, 'foo.parquet')
    offset = '2H'

    # Case 2 - Date data used for grouping is 'timestamp' column.    
    # Step 1 - Writing of a 1st df and testing correct grouping.
    # df1 is not sorted: it will be sorted before writing.
    df1 = pd.DataFrame({'timestamp': [pd.Timestamp('2020/01/02 01:59:00'),
                                      pd.Timestamp('2020/01/02 03:59:00'),
                                      pd.Timestamp('2020/01/02 02:59:00')],
                        'humidity': [0.3, 0.8, 0.9],
                        'pressure': [1e5, 1.1e5, 0.95e5],
                        'location': ['Paris', 'Paris', 'Milan']})
    write(fn, df1, row_group_offsets=offset, file_scheme='hive',
          date_col='timestamp', write_index=False)
    pf = ParquetFile(fn)
    expected_1_1 = pd.DataFrame({'timestamp': [pd.Timestamp('2020/01/02 01:59:00')],
                                 'humidity': [0.3],
                                 'pressure': [1e5],
                                 'location': ['Paris']})
    recorded_1_1 = pf.read_row_group_file(pf.row_groups[0], pf.columns, pf.categories)
    assert expected_1_1.equals(recorded_1_1)
    expected_1_2 = pd.DataFrame({'timestamp': [pd.Timestamp('2020/01/02 02:59:00'),
                                               pd.Timestamp('2020/01/02 03:59:00')],
                                 'humidity': [0.9, 0.8],
                                 'pressure': [0.95e5, 1.1e5],
                                 'location': ['Milan', 'Paris']})
    recorded_1_2 = pf.read_row_group_file(pf.row_groups[1], pf.columns, pf.categories)
    assert expected_1_2.equals(recorded_1_2)

    # Step 2 - Updating with a 2nd df having overlapping data and duplicates
    #          and testing correct row group insertion and removal of duplicates.
    # df2 is not sorted: it will be sorted before writing.
    df2 = pd.DataFrame({'timestamp': [pd.Timestamp('2020/01/01 01:59:00'),
                                      pd.Timestamp('2020/01/02 01:58:00'),
                                      pd.Timestamp('2020/01/02 01:59:00'),
                                      pd.Timestamp('2020/01/02 00:38:00'),
                                      pd.Timestamp('2020/01/03 02:59:00')],
                        'humidity': [0.5, 0.3, 0.4, 0.8, 1.1],
                        'pressure': [9e4, 1e5, 1.1e5, 1.1e5, 0.95e5],
                        'location': ['Tokyo', 'Paris', 'Paris', 'Paris', 'Paris']})
    write(fn, df2, row_group_offsets=offset, file_scheme='hive', append=True,
          drop_duplicates_on = 'timestamp', date_col='timestamp', write_index=False)
    pf = ParquetFile(fn)
    expected_2_1 = pd.DataFrame({'timestamp': [pd.Timestamp('2020/01/01 01:59:00')],
                                 'humidity': [0.5],
                                 'pressure': [9e4],
                                 'location': ['Tokyo']})
    recorded_2_1 = pf.read_row_group_file(pf.row_groups[0], pf.columns, pf.categories)
    assert expected_2_1.equals(recorded_2_1) # test row group insert in 1st position
    expected_2_2 = pd.DataFrame({'timestamp': [pd.Timestamp('2020-01-02 00:38:00'),
                                           pd.Timestamp('2020-01-02 01:58:00'),
                                           pd.Timestamp('2020-01-02 01:59:00')],
                                 'humidity': [0.8, 0.3, 0.4],
                                 'pressure': [1.1e5, 1e5, 1.1e5],
                                 'location': ['Paris','Paris','Paris']})
    recorded_2_2 = pf.read_row_group_file(pf.row_groups[1], pf.columns, pf.categories)
    assert expected_2_2.equals(recorded_2_2) # test drop duplicate according timestamp and keep last.
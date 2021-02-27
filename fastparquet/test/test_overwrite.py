"""
   test_overwrite.py
   Tests for overwriting parquet files.
"""

import pandas as pd
import pytest
from fastparquet import write, ParquetFile
from fastparquet.test.util import tempdir


def test_write_with_rgp_by_date_as_index(tempdir):

    # Step 1 - Writing of a 1st df, with `row_group_offsets=0`,
    # `file_scheme=hive` and `partition_on=['location', 'color`].
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9],
                        'pressure': [1e5, 1.1e5, 0.95e5],
                        'location': ['Paris', 'Paris', 'Milan'],
                        'color': ['red', 'black', 'blue']})
    write(tempdir, df1, row_group_offsets=0, file_scheme='hive',
          partition_on=['location', 'color'])

    # Step 2 - Overwriting with a 2nd df having overlapping data, in
    # 'overwrite' mode:
    # `row_group_offsets=0`, `file_scheme=hive`,
    # `partition_on=['location', 'color`] and `append=True`.
    df2 = pd.DataFrame({'humidity': [0.5, 0.3, 0.4, 0.8, 1.1],
                        'pressure': [9e4, 1e5, 1.1e5, 1.1e5, 0.95e5],
                        'location': ['Milan', 'Paris', 'Paris', 'Paris', 'Paris'],
                        'color': ['red', 'black', 'black', 'green', 'green' ]})

    write(tempdir, df2, row_group_offsets=0, file_scheme='hive', append='overwrite',
          partition_on=['location', 'color'])

    expected = pd.DataFrame({'humidity': [0.9, 0.5, 0.3, 0.4, 0.8, 1.1, 0.3],
                             'pressure': [9.5e4, 9e4, 1e5, 1.1e5, 1.1e5, 9.5e4, 1e5],
                             'location': ['Milan', 'Milan', 'Paris', 'Paris', 'Paris', 'Paris', 'Paris'],
                             'color': ['blue', 'red', 'black', 'black', 'green', 'green', 'red']})\
                           .astype({'location': 'category', 'color': 'category'})
    recorded = ParquetFile(tempdir).to_pandas()
    # df1 is 3 rows, df2 is 5 rows. Because of overlapping data with keys
    # 'location' = 'Paris' & 'color' = 'black' (1 row in df2, 2 rows in df2)
    # resulting df contains for this combination values of df2 and not that of
    # df1. Total resulting number of rows is 7.
    assert expected.equals(recorded)

def test_several_existing_parts_in_folder_exception(tempdir):

    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9, 0.7],
                        'pressure': [1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Paris', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'yes', 'yes']})

    write(tempdir, df1, row_group_offsets = 1, file_scheme='hive',
          write_index=False, partition_on=['location', 'exterior'])

    with pytest.raises(ValueError, match="^Some partition folders"):
        write(tempdir, df1, row_group_offsets = 0, file_scheme='hive',
              write_index=False, partition_on=['location', 'exterior'],
              append='overwrite')


"""
   test_overwrite.py
   Tests for overwriting parquet files.
"""
import os
import pytest

import pandas as pd

from fastparquet import write, ParquetFile
from fastparquet.writer import update
from .util import tempdir


def test_single_part_in_partitions(tempdir):
    # Step 1 - Writing of a 1st df, with `row_group_offsets=0`,
    # `file_scheme=hive` and `partition_on=['location', 'color`].
    # 'location/color' keys (used in this test for partitioning) are
    # intentionally in sorted order, as partitioning in fastparquet rely on
    # pandas groupby with std settings, which result in sorting keys for
    # grouping. This way, that this setting changes or not, this test case is
    # not impacted anyway.
    df1 = pd.DataFrame({'humidity': [0.9, 0.8, 0.93],
                        'pressure': [0.95e5, 1.1e5, 1e5],
                        'location': ['Milan', 'Paris', 'Paris'],
                        'color': ['blue', 'black', 'red']})
    write(tempdir, df1, row_group_offsets=0, file_scheme='hive',
          partition_on=['location', 'color'])

    # Step 2 - Overwriting with a 2nd df having overlapping data.
    df2 = pd.DataFrame({
                     'humidity': [0.5, 0.3, 0.4, 0.8, 1.1],
                     'pressure': [9e4, 1e5, 1.1e5, 1.1e5, 0.95e5],
                     'location': ['Milan', 'Paris', 'Paris', 'Paris', 'Paris'],
                     'color': ['red', 'black', 'black', 'green', 'green' ]})
    update(tempdir, df2, row_group_offsets=0)
    recorded = ParquetFile(tempdir).to_pandas()

    expected = pd.DataFrame({
   'humidity': [0.9, 0.3, 0.4, 0.93, 0.5, 0.8, 1.1],
   'pressure': [9.5e4, 1e5, 1.1e5, 1e5, 9e4, 1.1e5, 9.5e4],
   'location': ['Milan', 'Paris', 'Paris', 'Paris', 'Milan', 'Paris', 'Paris'],
   'color': ['blue', 'black', 'black', 'red', 'red', 'green', 'green']
                            })
    expected = expected.astype({'location': 'category', 'color': 'category'})
    # df1 is 3 rows, df2 is 5 rows. Because of overlapping data with keys
    # 'location' = 'Paris' & 'color' = 'black' (1 row in df1, 2 rows in df2)
    # resulting df contains for this combination:
    # - values from df2
    # - and not that of df1.
    # Total resulting number of rows is 7.
    assert expected.equals(recorded)


def test_multiple_parts_in_partitions(tempdir):
    # Several existing parts in partition 'Paris/yes'.
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9, 0.7],
                        'pressure': [1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Paris', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'yes', 'yes']})
    write(tempdir, df1, row_group_offsets=1, file_scheme='hive',
          write_index=False, partition_on=['location', 'exterior'])

    df2 = pd.DataFrame({'humidity': [0.4, 0.8, 0.9, 0.7],
                        'pressure': [1.1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Tokyo', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'no', 'yes']})
    update(tempdir, df2, row_group_offsets=1)

    expected = pd.DataFrame(
           {'humidity': [0.4, 0.7, 0.8, 0.9, 0.8, 0.9],
            'pressure': [1.1e5, 1e5, 1.1e5, 9.5e4, 1.1e5, 9.5e4],
            'location': ['Paris', 'Paris', 'Paris', 'Milan', 'Tokyo', 'Milan'],
            'exterior': ['yes', 'yes', 'no', 'yes', 'no', 'no']})\
                 .astype({'location': 'category', 'exterior': 'category'})
    recorded = ParquetFile(tempdir).to_pandas()
    assert expected.equals(recorded)


def test_with_actually_no_rg_to_overwrite(tempdir):
    # Step 1 - Writing of a 1st df, with `row_group_offsets=0`,
    # `file_scheme=hive` and `partition_on=['location', 'color`].
    df1 = pd.DataFrame({'humidity': [0.9, 0.8, 0.93],
                        'pressure': [0.95e5, 1.1e5, 1e5],
                        'location': ['Milan', 'Paris', 'Paris'],
                        'color': ['blue', 'black', 'red']})
    write(tempdir, df1, row_group_offsets=0, file_scheme='hive',
          partition_on=['location', 'color'])

    # Step 2 - 'Overwriting' with a 2nd df having actually no overlapping data.
    df2 = pd.DataFrame({
                     'humidity': [0.5, 0.3],
                     'pressure': [9e4, 1e5],
                     'location': ['Milan', 'Paris'],
                     'color': ['red', 'green']})
    update(tempdir, df2, row_group_offsets=0)

    expected = pd.DataFrame({
                     'humidity': [0.9, 0.8, 0.93, 0.5, 0.3],
                     'pressure': [9.5e4, 1.1e5, 1e5, 9e4, 1e5],
                     'location': ['Milan', 'Paris', 'Paris', 'Milan', 'Paris'],
                     'color': ['blue', 'black', 'red', 'red', 'green']})
    expected = expected.astype({'location': 'category', 'color': 'category'})
    recorded = ParquetFile(tempdir).to_pandas()
    assert expected.equals(recorded)


def test_multiple_parts_in_partitions_thru_write(tempdir):
    # Several existing parts in folder 'Paris/yes'.
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9, 0.7],
                        'pressure': [1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Paris', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'yes', 'yes']})
    write(tempdir, df1, row_group_offsets=1, file_scheme='hive',
          write_index=False, partition_on=['location', 'exterior'])

    df2 = pd.DataFrame({'humidity': [0.4, 0.8, 0.9, 0.7],
                        'pressure': [1.1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Tokyo', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'no', 'yes']})
    write(tempdir, df2, row_group_offsets=1, append='overwrite')

    expected = pd.DataFrame(
           {'humidity': [0.4, 0.7, 0.8, 0.9, 0.8, 0.9],
            'pressure': [1.1e5, 1e5, 1.1e5, 9.5e4, 1.1e5, 9.5e4],
            'location': ['Paris', 'Paris', 'Paris', 'Milan', 'Tokyo', 'Milan'],
            'exterior': ['yes', 'yes', 'no', 'yes', 'no', 'no']})\
                 .astype({'location': 'category', 'exterior': 'category'})
    recorded = ParquetFile(tempdir).to_pandas()
    assert expected.equals(recorded)


def test_no_partitioning_exception(tempdir):
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9, 0.7],
                        'pressure': [1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Paris', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'yes', 'yes']})
    # No partitions.
    write(tempdir, df1, row_group_offsets=1, file_scheme='hive',
          write_index=False)
    with pytest.raises(ValueError, match="^No partitioning"):
        update(tempdir, df1, row_group_offsets=0)


def test_simple_scheme_exception(tempdir):
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9, 0.7],
                        'pressure': [1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Paris', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'yes', 'yes']})
    # Simple file scheme.
    fn = os.path.join(tempdir, 'foo.parquet')
    write(fn, df1, row_group_offsets=1, file_scheme='simple',
          write_index=False)
    with pytest.raises(ValueError, match="^Not possible to overwrite"):
        update(fn, df1, row_group_offsets=0)


def test_multiple_parts_in_partitions_with_renaming(tempdir):
    # Several existing parts in partition 'Paris/yes'.
    df1 = pd.DataFrame({'humidity': [0.3, 0.8, 0.9, 0.7],
                        'pressure': [1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Paris', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'yes', 'yes']})
    write(tempdir, df1, row_group_offsets=1, file_scheme='hive',
          write_index=False, partition_on=['location', 'exterior'])

    df2 = pd.DataFrame({'humidity': [0.4, 0.8, 0.9, 0.7],
                        'pressure': [1.1e5, 1.1e5, 0.95e5, 1e5],
                        'location': ['Paris', 'Tokyo', 'Milan', 'Paris'],
                        'exterior': ['yes', 'no', 'no', 'yes']})
    # 'update' without file shuffling.
    update(tempdir, df2, row_group_offsets=1, sort_pnames=False)
    recorded = ParquetFile(tempdir)
    pnames_rec = [rg.columns[0].file_path for rg in recorded.row_groups]
    pnames_ref = ['location=Paris/exterior=yes/part.3.parquet',
                  'location=Paris/exterior=yes/part.6.parquet',
                  'location=Paris/exterior=no/part.1.parquet',
                  'location=Milan/exterior=yes/part.2.parquet',
                  'location=Tokyo/exterior=no/part.4.parquet',
                  'location=Milan/exterior=no/part.5.parquet']
    assert pnames_rec == pnames_ref
    # update' again with file shuffling.
    update(tempdir, df2, row_group_offsets=1, sort_pnames=True)
    recorded = ParquetFile(tempdir)
    pnames_rec = [rg.columns[0].file_path for rg in recorded.row_groups]
    pnames_ref = ['location=Paris/exterior=yes/part.0.parquet',
                  'location=Paris/exterior=yes/part.1.parquet',
                  'location=Paris/exterior=no/part.2.parquet',
                  'location=Milan/exterior=yes/part.3.parquet',
                  'location=Tokyo/exterior=no/part.4.parquet',
                  'location=Milan/exterior=no/part.5.parquet']
    assert pnames_rec == pnames_ref

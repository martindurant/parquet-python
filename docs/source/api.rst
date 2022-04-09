API
===

.. currentmodule:: fastparquet

.. autosummary::
    ParquetFile
    ParquetFile.to_pandas
    ParquetFile.iter_row_groups
    ParquetFile.info
    ParquetFile.head
    ParquetFile.count
    ParquetFile.__getitem__
    write

.. autoclass:: ParquetFile
    :members:

.. automethod:: ParquetFile.to_pandas

.. automethod:: ParquetFile.iter_row_groups

.. autofunction:: write

.. autofunction:: update_file_custom_metadata

Backend File-systems
====================

Fastparquet can use alternatives to the local disk for reading and writing parquet.

One example of such a backend file-system is `s3fs <http://s3fs.readthedocs.io>`_, to connect to
AWS's S3 storage. In the following, the login credentials are automatically inferred from the system
(could be environment variables, or one of several possible configuration files).

.. code-block:: python

    import s3fs
    from fastparquet import ParquetFile
    s3 = s3fs.S3FileSystem()
    myopen = s3.open
    pf = ParquetFile('/mybucket/data.parquet', open_with=myopen)
    df = pf.to_pandas()

The function ``myopen`` provided to the constructor must be callable with ``f(path, mode)``
and produce an open file context.

The resultant ``pf`` object is the same as would be generated locally, and only requires a relatively short
read from the remote store. If '/mybucket/data.parquet' contains a sub-key called "_metadata", it will be
read in preference, and the data-set is assumed to be multi-file.


Similarly, providing an open function and another to make any necessary directories (only necessary in multi-file mode), we can write to the s3 file-system:

.. code-block:: python

   write('/mybucket/output_parq', data, file_scheme='hive',
         row_group_offsets=[0, 500], open_with=myopen, mkdirs=noop)

(In the case of s3, no intermediate directories need to be created)

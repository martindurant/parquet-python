fastparquet
===========

A python interface to the parquet file format.

Introduction
------------

The `Parquet format <https://github.com/Parquet/parquet-format>`_ is a common binary data store, used
particularly in the Hadoop/big-data sphere. It provides several advantages relevant to big-data
processing:

- columnar storage, only read the data of interest
- efficient binary packing
- choice of compression algorithms and encoding
- split data into files, allowing for parallel processing
- range of logical types
- statistics stored in metadata allow for skipping unneeded chunks
- data partitioning using the directory structure

Since it was developed as part of the Hadoop ecosystem, parquet's reference implementation is
written in java. This package aims to provide a performant library to read and write parquet files
from python, without any need for a python-java bridge. This will make the parquet format an
ideal storage mechanism for python-based big data workflows.

The tabular nature of Parquet is a good fit for the pandas data-frame objects, and
we exclusively deal with data-frame<->parquet.

Highlights
----------

The original outline plan for this project can be found `here <https://github.com/dask/fastparquet/issues/1>`_

Briefly, some features of interest:

- read and write parquet files, in single- or multiple-file format. The latter is common found in hive/spark usage.
- choice of encoding and compression per-column; ability to choose row divisions and partitioning on write.
- acceleration of both reading and writing using `numba <http://numba.pydata.org/>`_
- ability to read and write to arbitrary file-like objects, allowing interoperability with `s3fs <http://s3fs.readthedocs.io/>`_, `hdfs3 <http://hdfs3.readthedocs.io/>`_, `adlfs <https://github.com/Azure/azure-data-lake-store-python>`_ and possibly others.
- can be called from `dask <http://dask.pydata.org>`_, to enable parallel reading and writing with parquet files, possibly distributed across a cluster.

Caveats, Known Issues
---------------------

Not all parts of the parquet-format have been implemented yet or tested.
fastparquet is, however, capable of reading all the data files from the
`parquet-compatability <https://github.com/Parquet/parquet-compatibility>`_
project.

A list of current issues can be found `here <https://github.com/dask/fastparquet/>`_.



Index
-----

.. toctree::

    install
    quickstart
    details
    api
    filesystems
    developer

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

fastparquet
===========

A Python interface to the Parquet file format.

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

Since it was developed as part of the Hadoop ecosystem, Parquet's reference implementation is
written in Java. This package aims to provide a performant library to read and write Parquet files
from Python, without any need for a Python-Java bridge. This will make the Parquet format an
ideal storage mechanism for Python-based big data workflows.

The tabular nature of Parquet is a good fit for the Pandas data-frame objects, and
we exclusively deal with data-frame<->Parquet.

Highlights
----------

The original outline plan for this project can be found `here <https://github.com/dask/fastparquet/issues/1>`_

Briefly, some features of interest:

- read and write Parquet files, in single- or multiple-file format. The latter is commonly found in hive/Spark usage.
- choice of compression per-column and various optimized encoding schemes; ability to choose row divisions and partitioning on write.
- acceleration of both reading and writing using `numba <http://numba.pydata.org/>`_
- ability to read and write to arbitrary file-like objects, allowing interoperability with `s3fs <http://s3fs.readthedocs.io/>`_, `hdfs3 <http://hdfs3.readthedocs.io/>`_, `adlfs <https://github.com/Azure/azure-data-lake-store-python>`_ and possibly others.
- can be called from `dask <http://dask.pydata.org>`_, to enable parallel reading and writing with Parquet files, possibly distributed across a cluster.

Caveats, Known Issues
---------------------

Not all parts of the Parquet-format have been implemented yet or tested.
fastparquet is, however, capable of reading all the data files from the
`parquet-compatibility <https://github.com/Parquet/parquet-compatibility>`_
project. Some encoding mechanisms in Parquet are rare, and may be implemented
on request - please post an issue.

Some deeply-nested columns will not be readable, e.g., lists of lists.

Not all output options will be compatible with every other Parquet
framework, which each implement only a subset of the standard, see
the usage notes.

A list of current issues can be found `here <https://github.com/dask/fastparquet/>`_.

Relation to Other Projects
--------------------------

- `parquet-python <https://github.com/jcrobak/parquet-python>`_ is the original
pure-Python Parquet quick-look utility which was the inspiration for fastparquet.

-  `parquet-cpp <https://github.com/apache/parquet-cpp/>`_ is a low-level C++
implementation of the Parquet format which can be called from Python using
Apache `Arrow <http://pyarrow.readthedocs.io/en/latest/>`_ bindings.
Future collaboration with parquet-cpp is possible, in the medium term, and that
perhaps their low-level
routines will replace some functions in fastparquet or that high-level logic in
fastparquet will be migrated to C++.

- `PySpark <http://Spark.apache.org/docs/2.1.0/programming-guide.html>`_, a Python API to the Spark
engine, interfaces Python commands with a Java/Scala execution core, and thereby
gives Python programmers access to the Parquet format. fastparquet has no
defined relationship to PySpark, but can provide an alternative path to providing
data to Spark or reading data produced by Spark without invoking a PySpark client
or interacting directly with the scheduler.

- fastparquet lives within the `dask <http://dask.pydata.org>`_ ecosystem, and
although it is useful by itself, it is designed to work well with dask for parallel
execution, as well as related libraries such as s3fs for pythonic access to
Amazon S3.


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

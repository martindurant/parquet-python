Installation
============

Requirements
------------

Required:

#. numpy
#. pandas
#. cramjam
#. thrift

`cramjam`_ provides compression codecs: gzip, snappy, lz4, brotli, zstd

.. _cramjam: https://github.com/milesgranger/pyrus-cramjam

Optional compression codec:

#. python-lzo/lzo

Dev requirements
----------------
To run all of the tests, you will need the following, in addition to the requirements above:

#. python=3.8
#. bson
#. lz4
#. lzo
#. pytest
#. dask
#. pytest-cov
#. pyspark

Some of these (e.g., pyspark) are optional and will result in skipped tests if not present.

Installation
------------

Install using conda::

   conda install -c conda-forge fastparquet

install from PyPI::

   pip install fastparquet

or install latest version from github::

   pip install git+https://github.com/dask/fastparquet

Please be sure to install numpy before fastparquet when using pip, as pip sometimes
can fail to solve the environment.


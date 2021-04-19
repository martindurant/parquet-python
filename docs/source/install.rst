Installation
============

Requirements
------------

Required:

- numba
- numpy
- pandas
- pytest
- cramjam

`cramjam`_ provides compression codecs: gzip, snappy, lz4, brotli, zstd

.. _cramjam: https://github.com/milesgranger/pyrus-cramjam

Optional compression codec:

- python-lzo

Installation
------------

Install using conda::

   conda install -c conda-forge fastparquet

install from pypi::

   pip install fastparquet

or install latest version from github::

   pip install git+https://github.com/dask/fastparquet

For the pip methods, numba must have been previously installed (using conda, or from source).


Installation
============

Requirements
------------

Required:

- numba
- numpy
- pandas
- pytest

Optional (compression algorithms; gzip is always available):

- python-snappy
- python-lzo
- brotli

Installation
------------

Install using conda::

   conda install -c conda-forge fastparquet

install from pypi::

   pip install fastparquet

or install latest version from github::

   pip install git+https://github.com/dask/fastparquet

For the pip methods, numba must have been previously installed (using conda, or from source).


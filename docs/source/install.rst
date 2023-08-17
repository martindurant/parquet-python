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

Installation
------------

Install using conda::

   conda install -c conda-forge fastparquet

install from PyPI::

   pip install fastparquet

or install latest version from github, "main" branch::

   pip install git+https://github.com/dask/fastparquet

Please be sure to install numpy before fastparquet when using pip, as pip sometimes
can fail to solve the environment. Depending on your platform, pip may pull binary wheels
or attempt to rebuild fastparquet.

Dev requirements
----------------
To run all of the tests, you will need the following, in addition to the requirements above:

#. python=3.8
#. bson
#. lz4
#. lzo
#. pytest
#. dask
#. moto/s3fs
#. pytest-cov
#. pyspark

Some of these (e.g., pyspark) are optional and will result in skipped tests if not present.

Tests use pytest.

**Building Docs**

The *docs/* directory contains source code for the documentation. You will
need sphinx and numpydoc to successfully build. sphinx allows output in
many formats, including html:

.. code-block:: bash

    # in directory docs/
    make html

This will produce a ``build/html/`` subdirectory, where the entry point is
``index.html``.




.. raw:: html

    <script data-goatcounter="https://distdatacats.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>

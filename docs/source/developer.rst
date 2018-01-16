Developer Guide
===============

Fastparquet is a free and open-source project.
We welcome contributions in the form of bug reports, documentation, code, design proposals, and more.
This page provides resources on how best to contribute.

**Bug reports**

Please file an issue on `github <https://github.com/dask/fastparquet/>`_.

**Running tests**

Aside from the requirements for using this package, the following additional
packages should be present:

.. code-block:: bash

    pip install pytest

Some tests also require:

- s3fs
- moto
- pyspark

***Build `speedups.pyx`***

The speedups module was written in cython and must be compiled:

.. code-block:: bash

    python setup.py build_ext --inplace

***Install `python-snappy`***

The `python-snappy` is required for tests, but it requires a compiler,
which can be difficult to get running on Windows. Instead, [download an
unofficial pre-compiled version](https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-snappy)
and install.

.. code-block:: bash

    pip install python_snappy-0.5.1-cp27-cp27m-win32.whl

**Building Docs**

The *docs/* directory contains source code for the documentation. You will
need sphinx and numpydoc to successfully build. sphinx allows output in
many formats, including html:

.. code-block:: bash

    # in directory docs/
    make html

This will produce a ``build/html/`` subdirectory, where the entry point is
``index.html``.

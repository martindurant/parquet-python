"""setup.py - build script for parquet-python."""

import fnmatch
import os
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as build_py_orig


class build_ext(_build_ext):
    # Kudos to https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py/21621689
    def finalize_options(self):
        if sys.version_info[0] >= 3:
            import builtins
        else:
            import __builtin__ as builtins
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


allowed = ('--help-commands', '--version', 'egg_info', 'clean')
if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1] in allowed):
    # NumPy and cython are not required for these actions. They must succeed
    # so pip can install fastparquet when these requirements are not available.
    extra = {}
else:
    modules_to_build = {
        'fastparquet.speedups': ['fastparquet/speedups.pyx']
    }
    try:
        from Cython.Build import cythonize
        def fix_exts(sources):
            return sources
    except ImportError:
        def cythonize(modules, language_level):
            return modules
        def fix_exts(sources):
            return [s.replace('.pyx', '.c') for s in sources]

    modules = [
        Extension(mod, fix_exts(sources))
        for mod, sources in modules_to_build.items()]
    extra = {'ext_modules': cythonize(modules, language_level=3)}

install_requires = open('requirements.txt').read().strip().split('\n')

setup(
    name='fastparquet',
    version='0.4.2',
    description='Python support for Parquet file format',
    author='Martin Durant',
    author_email='mdurant@continuum.io',
    url='https://github.com/dask/fastparquet/',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=['fastparquet'],
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    setup_requires=[
        'pytest-runner',
    ] + [p for p in install_requires if p.startswith('numpy')],
    extras_require={
        'brotli': ['brotli'],
        'lz4': ['lz4 >= 0.19.1'],
        'lzo': ['python-lzo'],
        'snappy': ['python-snappy'],
        'zstandard': ['zstandard'],
        'zstd': ['zstd'],
    },
    tests_require=[
        'pytest',
        'python-snappy',
        'lz4 >= 0.19.1',
        'zstandard',
        'zstd',
    ],
    long_description=(open('README.rst').read() if os.path.exists('README.rst')
                      else ''),
    package_data={'fastparquet': ['*.thrift']},
    include_package_data=True,
    exclude_package_data={'fastparquet': ['test/*']},
    python_requires=">=3.6,",
    **extra
)

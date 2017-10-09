"""setup.py - build script for parquet-python."""

import os
import sys
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

allowed = ('--help-commands', '--version', 'egg_info', 'clean')
if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1] in allowed):
    # NumPy and cython are not required for these actions. They must succeed
    # so pip can install fastparquet when these requirements are not available.
    extra = {}
else:
    import numpy as np
    from Cython.Build import cythonize
    cython_modules = [Extension('fastparquet.speedups',
                                ['fastparquet/speedups.pyx'],
                                include_dirs=[np.get_include()])]
    extra = {'ext_modules': cythonize(cython_modules)}

setup(
    name='fastparquet',
    version='0.1.3',
    description='Python support for Parquet file format',
    author='Martin Durant',
    author_email='mdurant@continuum.io',
    url='https://github.com/martindurant/fastparquet/',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=['fastparquet'],
    install_requires=[open('requirements.txt').read().strip().split('\n')],
    long_description=(open('README.rst').read() if os.path.exists('README.rst')
                      else ''),
    package_data={'fastparquet': ['*.thrift']},
    include_package_data=True,
    **extra
)

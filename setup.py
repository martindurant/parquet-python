"""setup.py - build script for parquet-python."""

import os
import sys
try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext as _build_ext

# Kudos to https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py/21621689
class build_ext(_build_ext):
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
        def cythonize(modules):
            return modules
        def fix_exts(sources):
            return [s.replace('.pyx', '.c') for s in sources]

    modules = [
        Extension(mod, fix_exts(sources))
        for mod, sources in modules_to_build.items()]

    extra = {'ext_modules': cythonize(modules)}

install_requires = open('requirements.txt').read().strip().split('\n')

# add pytest-runner from setup_requires
install_requires.append('pytest-runner')

setup(
    name='fastparquet',
    version='0.2.1',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    packages=['fastparquet'],
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    # remove wierd setup_requires to get rid of easy_install
    # setup_requires=[
    #    'pytest-runner',
    #    [p for p in install_requires if p.startswith('numpy')][0]
    #],
    extras_require={
        'brotli': ['brotli'],
        'lz4': ['lz4 >= 0.19.1'],
        'lzo': ['python-lzo'],
        'snappy': ['python-snappy'],
        'zstandard': ['zstandard']
    },
    tests_require=[
        'pytest',
        'python-snappy',
        'lz4 >= 0.19.1',
        'zstandard',
    ],
    long_description=(open('README.rst').read() if os.path.exists('README.rst')
                      else ''),
    package_data={'fastparquet': ['*.thrift']},
    include_package_data=True,
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*,",
    **extra
)

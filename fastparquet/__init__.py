"""parquet - read parquet files."""
__version__ = "0.7.2"

from .thrift_structures import parquet_thrift
from .core import read_thrift
from .writer import write
from . import core, schema, converted_types, api
from .api import ParquetFile
from .util import ParquetException

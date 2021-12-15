"""parquet - read parquet files."""
__version__ = "0.7.2"

from .writer import write
from . import core, schema, converted_types, api
from .api import ParquetFile
from .util import ParquetException

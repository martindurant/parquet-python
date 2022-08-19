"""parquet - read parquet files."""
__version__ = "0.8.2"

from .writer import write, update_file_custom_metadata
from . import core, schema, converted_types, api
from .api import ParquetFile
from .util import ParquetException

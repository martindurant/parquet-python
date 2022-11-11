"""parquet - read parquet files."""

from .writer import write, update_file_custom_metadata
from . import core, schema, converted_types, api
from .api import ParquetFile
from .util import ParquetException

from ._version import __version__

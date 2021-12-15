from functools import partial
from .parquet.ttypes import *


def __getattr__(name):
    # for compatability with coe that calls, e.g., parquet_thrift.RowGroup(...)
    from ..cencoding import ThriftObject
    return partial(ThriftObject.from_fields, thrift_name=name)

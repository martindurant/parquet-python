"""encoding.py - methods for reading parquet encoded data blocks."""
import numpy as np
from fastparquet.cencoding import read_bitpacked1, NumpyIO
from fastparquet import parquet_thrift


def read_plain_boolean(raw_bytes, count, out=None):
    data = np.frombuffer(raw_bytes, dtype='uint8')
    out = out or np.empty(count, dtype=bool)
    read_bitpacked1(NumpyIO(data), count, NumpyIO(out.view('uint8')))
    return out[:count]


DECODE_TYPEMAP = {
    parquet_thrift.Type.INT32: np.int32,
    parquet_thrift.Type.INT64: np.int64,
    parquet_thrift.Type.INT96: np.dtype('S12'),
    parquet_thrift.Type.FLOAT: np.float32,
    parquet_thrift.Type.DOUBLE: np.float64,
}


def read_plain(raw_bytes, type_, count, width=0, utf=False, stat=False):
    if type_ in DECODE_TYPEMAP:
        dtype = DECODE_TYPEMAP[type_]
        return np.frombuffer(memoryview(raw_bytes), dtype=dtype, count=count)
    if type_ == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
        if count == 1:
            width = len(raw_bytes)
        dtype = np.dtype('S%i' % width)
        return np.frombuffer(memoryview(raw_bytes), dtype=dtype, count=count)
    if type_ == parquet_thrift.Type.BOOLEAN:
        return read_plain_boolean(raw_bytes, count)
    if type_ == parquet_thrift.Type.BYTE_ARRAY:
        if stat:
            if utf:
                return np.array([bytes(raw_bytes).decode()], dtype='O')
            else:
                return np.array([bytes(raw_bytes)], dtype='O')
        return unpack_byte_array(raw_bytes, count, utf=utf)


def byte_stream_unsplit8(arr):
    assert arr.dtype == "f8"
    out = np.empty_like(arr)
    view1 = out.view('uint8')
    view2 = arr.view("uint8")
    l = len(arr)
    for i in range(8):
        view1[i::8] = view2[i*l: (i+1)*l]
    return out


def byte_stream_split8(arr):
    assert arr.dtype == "f8"
    out = np.empty_like(arr)
    view1 = out.view('uint8')
    view2 = arr.view("uint8")
    l = len(arr)
    for i in range(8):
        view1[i*l: (i+1)*l] = view2[i::8]
    return out



def byte_stream_unsplit4(arr):
    assert arr.dtype == "f4"
    out = np.empty_like(arr)
    view1 = out.view('uint8')
    view2 = arr.view("uint8")
    l = len(arr)
    for i in range(4):
        view1[i::4] = view2[i*l: (i+1)*l]
    return out


def byte_stream_split4(arr):
    assert arr.dtype == "f4"
    out = np.empty_like(arr)
    view1 = out.view('uint8')
    view2 = arr.view("uint8")
    l = len(arr)
    for i in range(4):
        view1[i*l: (i+1)*l] = view2[i::4]
    return out

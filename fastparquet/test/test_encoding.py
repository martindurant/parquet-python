"""test_encoding.py - tests for deserializing parquet data."""
import numpy as np
import struct

import fastparquet.encoding
from fastparquet import cencoding
from fastparquet import parquet_thrift


def test_int32():
    """Test reading bytes containing int32 data."""
    assert 999 == fastparquet.encoding.read_plain(
             struct.pack(b"<i", 999),
             parquet_thrift.Type.INT32, 1)


def test_int64():
    """Test reading bytes containing int64 data."""
    assert 999 == fastparquet.encoding.read_plain(
             struct.pack(b"<q", 999),
             parquet_thrift.Type.INT64, 1)


def test_int96():
    """Test reading bytes containing int96 data."""
    assert b'\x00\x00\x00\x00\x00\x00\x00\x00\xe7\x03\x00\x00' == fastparquet.encoding.read_plain(
             struct.pack(b"<qi", 0, 999),
             parquet_thrift.Type.INT96, 1)


def test_float():
    """Test reading bytes containing float data."""
    assert (9.99 - fastparquet.encoding.read_plain(
             struct.pack(b"<f", 9.99),
             parquet_thrift.Type.FLOAT, 1)) < 0.01


def test_double():
    """Test reading bytes containing double data."""
    assert (9.99 - fastparquet.encoding.read_plain(
             struct.pack(b"<d", 9.99),
             parquet_thrift.Type.DOUBLE, 1)) < 0.01


def test_fixed():
    """Test reading bytes containing fixed bytes data."""
    data = b"foobar"
    assert data[:3] == fastparquet.encoding.read_plain(
            data, parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY, -1, 3)[0]
    assert data[3:] == fastparquet.encoding.read_plain(
            data, parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY, -1, 3)[1]


def test_boolean():
    """Test reading bytes containing boolean data."""
    data = 0b1101
    d = struct.pack(b"<i", data)
    assert ([True, False, True, True] == fastparquet.encoding.read_plain(
            d, parquet_thrift.Type.BOOLEAN, 4)).all(0)


def testFourByteValue():
    """Test reading a run with a single four-byte value."""
    buf = np.frombuffer(struct.pack(b"<i", 1 << 30), np.uint8)
    fo = cencoding.NumpyIO(buf)
    out = np.zeros(10, np.uint32)
    o = cencoding.NumpyIO(out.view("uint8"))
    cencoding.read_rle(fo, 2 << 1, 30, o)
    assert ([1 << 30] * 2 == out[:2]).all()
    assert o.tell() == 8


def testSingleByte():
    """Test reading a single byte value."""
    buf = np.frombuffer(struct.pack(b"<i", 0x7F), np.uint8)
    fo = cencoding.NumpyIO(buf)
    out = cencoding.read_unsigned_var_int(fo)
    assert 0x7F == out
    assert fo.tell() == 1


def testFourByte():
    """Test reading a four byte value."""
    buf = np.frombuffer(struct.pack(b"<BBBB", 0xFF, 0xFF, 0xFF, 0x7F), np.uint8)
    fo = cencoding.NumpyIO(buf)
    out = cencoding.read_unsigned_var_int(fo)
    assert 0x0FFFFFFF == out
    assert fo.tell() == 4


def testFromExample():
    """Test a simple example."""
    raw_data_in = [0b10001000, 0b11000110, 0b11111010]
    encoded_bitstring = b'\x88\xc6\xfa'
    buf = np.frombuffer(encoded_bitstring, np.uint8)
    fo = cencoding.NumpyIO(buf)
    count = 8
    header = (count // 8) << 1
    out = np.empty(count, np.uint32)
    o = cencoding.NumpyIO(out.view("uint8"))
    cencoding.read_bitpacked(fo, header, 3, o)
    assert (list(range(8)) == out[:count]).all()
    assert fo.tell() == 3
    assert o.tell() == count * 4


def testWidths():
    """Test all possible widths for a single byte."""
    assert 0 == cencoding.width_from_max_int(0)
    assert 1 == cencoding.width_from_max_int(1)
    assert 2 == cencoding.width_from_max_int(2)
    assert 2 == cencoding.width_from_max_int(3)
    assert 3 == cencoding.width_from_max_int(4)
    assert 3 == cencoding.width_from_max_int(5)
    assert 3 == cencoding.width_from_max_int(6)
    assert 3 == cencoding.width_from_max_int(7)
    assert 4 == cencoding.width_from_max_int(8)
    assert 4 == cencoding.width_from_max_int(15)
    assert 5 == cencoding.width_from_max_int(16)
    assert 5 == cencoding.width_from_max_int(31)
    assert 6 == cencoding.width_from_max_int(32)
    assert 6 == cencoding.width_from_max_int(63)
    assert 7 == cencoding.width_from_max_int(64)
    assert 7 == cencoding.width_from_max_int(127)
    assert 8 == cencoding.width_from_max_int(128)
    assert 8 == cencoding.width_from_max_int(255)


def zigzag(n):
    return (n << 1) ^ (n >> 63)


def test_delta_from_def_1():
    # https://github.com/apache/parquet-format/blob/master/Encodings.md#example-1
    expected = [1, 2, 3, 4, 5]
    data = np.zeros(1000, dtype="uint8")
    o = cencoding.NumpyIO(data)
    cencoding.encode_unsigned_varint(8, o)  # block size
    cencoding.encode_unsigned_varint(1, o)  # miniblock count
    cencoding.encode_unsigned_varint(8, o)  # value count
    cencoding.encode_unsigned_varint(zigzag(1), o)  # first value (zigzag)

    # one and only miniblock
    cencoding.encode_unsigned_varint(zigzag(1), o)  # minimum delta (zigzag)
    o.write_byte(0)  # bit-width list (only one)

    o.seek(0)

    outdata = np.zeros(1000, dtype="int32")
    out = cencoding.NumpyIO(outdata.view("uint8"))
    cencoding.delta_binary_unpack(o, out)
    assert outdata[:5].tolist() == expected


def test_delta_from_def_2():
    # https://github.com/apache/parquet-format/blob/master/Encodings.md#example-2
    expected = [7, 5, 3, 1, 2, 3, 4, 5]
    data = np.zeros(1000, dtype="uint8")
    o = cencoding.NumpyIO(data)
    cencoding.encode_unsigned_varint(8, o)  # block size
    cencoding.encode_unsigned_varint(1, o)  # miniblock count
    cencoding.encode_unsigned_varint(8, o)  # value count
    cencoding.encode_unsigned_varint(zigzag(7), o)  # first value (zigzag)

    # one and only miniblock
    cencoding.encode_unsigned_varint(zigzag(-2), o)  # minimum delta (zigzag)
    o.write_byte(2)  # bit-width list (only one)
    o.write_byte(0b00000011)  # [0, 0, 0, 3]
    o.write_byte(0b11111100)  # [3, 3, 3, pad]

    o.seek(0)

    outdata = np.zeros(1000, dtype="int32")
    out = cencoding.NumpyIO(outdata.view("uint8"))
    cencoding.delta_binary_unpack(o, out)
    assert outdata[:8].tolist() == expected

import io
import numpy as np
import os
from fastparquet import encoding, core, ParquetFile, schema, util
import fastparquet.cencoding as encoding

TEST_DATA = 'test-data'
count = 1000


def test_read_bitpacked():
    results = np.empty(1000000, dtype=np.int32)
    with open(os.path.join(TEST_DATA, 'bitpack')) as f:
        for counter, l in enumerate(f):
            if counter > count:
                break
            raw, head, wid, res = eval(l)
            i = encoding.NumpyIO(raw)
            o = encoding.NumpyIO(results.view('uint8'))
            encoding.read_bitpacked(i, head, wid, o, itemsize=4)
            assert (res == results[:len(res)]).all()
            assert o.tell() == len(res) * 4


def test_read_bitpacked_uint():
    results = np.empty(1000000, dtype=np.uint8)
    with open(os.path.join(TEST_DATA, 'bitpack')) as f:
        for counter, l in enumerate(f):
            if counter > count:
                break
            raw, head, wid, res = eval(l)
            if wid > 7:
                continue
            print(wid)
            i = encoding.NumpyIO(raw)
            o = encoding.NumpyIO(results)
            encoding.read_bitpacked(i, head, wid, o, itemsize=1)
            assert (res == results[:len(res)]).all()
            assert o.tell() == len(res)


def test_rle():
    results = np.empty(1000000, dtype=np.int32)
    with open(os.path.join(TEST_DATA, 'rle')) as f:
        for i, l in enumerate(f):
            if i > count:
                break
            data, head, width, res = eval(l)
            i = encoding.NumpyIO(data)
            o = encoding.NumpyIO(results.view('uint8'))
            encoding.read_rle(i, head, width, o, itemsize=4)
            out = np.frombuffer(o.so_far(), dtype='int32')
            assert (res == out).all()


def test_uvarint():
    with open(os.path.join(TEST_DATA, 'uvarint')) as f:
        for i, l in enumerate(f):
            if i > count:
                break
            data, res = eval(l)
            i = encoding.NumpyIO(data)
            o = encoding.read_unsigned_var_int(i)
            assert (res == o)


def test_hybrid():
    results = np.empty(1000000, dtype=np.int32)
    with open(os.path.join(TEST_DATA, 'hybrid')) as f:
        for counter, l in enumerate(f):
            if counter > count // 20:
                break
            (data, width, length, res) = eval(l)
            i = encoding.NumpyIO(data)
            o = encoding.NumpyIO(results.view('uint8'))
            encoding.read_rle_bit_packed_hybrid(i, width, length or 0, o, itemsize=4)
            out = np.frombuffer(o.so_far(), dtype="int32")
            cond = (res == out).all()
            assert cond


def test_hybrid_extra_bytes():
    results = np.empty(1000000, dtype=np.int32)
    with open(os.path.join(TEST_DATA, 'hybrid')) as f:
        for i, l in enumerate(f):
            if i > count // 20:
                break
            (data, width, length, res) = eval(l)
            if length is not None:
                data2 = data + b'extra bytes'
            else:
                continue
            i = encoding.NumpyIO(data2)
            o = encoding.NumpyIO(results.view("uint8"))
            encoding.read_rle_bit_packed_hybrid(i, width, length, o, itemsize=4)
            out = np.frombuffer(o.so_far(), dtype="int32")
            cond = (res == out).all()
            assert cond
            assert i.tell() == len(data)


def test_read_data():
    with open(os.path.join(TEST_DATA, 'read_data')) as f:
        for i, l in enumerate(f):
            (data, fo_encoding, value_count, bit_width, res) = eval(l)
            i = encoding.NumpyIO(data)
            out = core.read_data(i, fo_encoding, value_count,
                                 bit_width)
            for o, r in zip(out, res):
                # result from old version is sometimes 1 value too long
                assert o == r


def test_to_pandas():
    fname = TEST_DATA+'/airlines_parquet/4345e5eef217aa1b-c8f16177f35fd983_1150363067_data.1.parq'
    pf = ParquetFile(fname)
    out = pf.to_pandas()
    assert len(out.columns) == 29
    # test for bad integer conversion
    assert (out.dep_time < 0).sum() == 0
    assert out.dep_time.dtype == 'float64'

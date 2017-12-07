# -*- coding: utf-8 -*-

import struct

from fastparquet.speedups import encode, decode

strings = [u"abc", u"a\x00c", u"héhé", u"プログラミング"]
bytess = [s.encode('utf8') for s in strings]


def test_roundtrip_bytes():
    out = decode(encode(bytess, utf8=False), len(bytess), utf8=False).tolist()
    assert out == bytess


def test_encode_bytes():
    expected = b''.join([(struct.pack('<i', len(s)) + s) for s in bytess])
    result = encode(bytess, utf8=False)
    assert expected == result


def test_decode_bytes():
    data = b''.join([(struct.pack('<i', len(s)) + s) for s in bytess])
    out = decode(data, len(bytess), utf8=False).tolist()
    assert out == bytess


def test_roundtrip_utf():
    out = decode(encode(strings, utf8=True), len(strings), utf8=True).tolist()
    assert out == strings


def test_encode_utf():
    expected = b''.join([(struct.pack('<i', len(s.encode('utf8'))) +
                          s.encode('utf8')) for s in strings])
    result = encode(strings, utf8=True)
    assert expected == result


def test_decode_utf():
    data = b''.join([(struct.pack('<i', len(s.encode('utf8'))) +
                      s.encode('utf8')) for s in strings])
    out = decode(data, len(strings), utf8=True).tolist()
    assert out == strings

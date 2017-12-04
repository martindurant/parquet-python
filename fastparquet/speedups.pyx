"""
Native accelerators for Parquet encoding and decoding.
"""

from __future__ import absolute_import

cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t n)

from cpython cimport (PyUnicode_AsUTF8String, PyUnicode_DecodeUTF8,
                      PyBytes_CheckExact, PyBytes_FromStringAndSize,
                      PyBytes_GET_SIZE, PyBytes_AS_STRING)

import numpy as np
cimport numpy as np


_obj_dtype = np.dtype('object')


def _to_array(obj):
    """
    Convert *obj* (a ndarray-compatible object, e.g. pandas Series)
    to a true ndarray.
    """
    if not isinstance(obj, np.ndarray):
        try:
            obj = obj.__array__()
        except AttributeError:
            raise TypeError("expected an ndarray-compatible object, got %r"
                            % (type(obj,)))
        assert isinstance(obj, np.ndarray)
    return obj


def _check_1d_object_array(np.ndarray arr):
    if arr.ndim != 1:
        raise TypeError("expected a 1d array")
    if arr.dtype != _obj_dtype:
        raise TypeError("expected an object array")


def array_encode_utf8(inp):
    """
    utf-8 encode all elements of a 1d ndarray of "object" dtype.
    A new ndarray of bytes objects is returned.
    """
    cdef:
        Py_ssize_t i, n
        np.ndarray[object] arr
        np.ndarray[object] result

    arr = _to_array(inp)
    _check_1d_object_array(arr)

    n = len(arr)
    result = np.empty(n, dtype=object)
    for i in range(n):
        # Fast utf-8 encoding, avoiding method call and codec lookup indirection
        result[i] = PyUnicode_AsUTF8String(arr[i])

    return result


def array_decode_utf8(inp):
    """
    utf-8 decode all elements of a 1d ndarray of "object" dtype.
    A new ndarray of unicode objects is returned.
    """
    cdef:
        Py_ssize_t i, n
        np.ndarray[object] arr
        np.ndarray[object] result
        object val

    arr = _to_array(inp)
    _check_1d_object_array(arr)

    n = len(arr)
    result = np.empty(n, dtype=object)
    for i in range(n):
        val = arr[i]
        if not PyBytes_CheckExact(val):
            raise TypeError("expected array of bytes")
        # Fast utf-8 decoding, avoiding method call and codec lookup indirection
        result[i] = PyUnicode_DecodeUTF8(
            PyBytes_AS_STRING(val),
            PyBytes_GET_SIZE(val),
            NULL,   # errors
            )

    return result


def pack_byte_array(list items):
    """
    Pack a variable length byte array column.
    A bytes object is returned.
    """
    cdef:
        Py_ssize_t i, n, itemlen, total_size
        unsigned char *start
        unsigned char *data
        object val, out

    # Strategy: compute the total output size and allocate it in one go.
    n = len(items)
    total_size = 0
    for i in range(n):
        val = items[i]
        if not PyBytes_CheckExact(val):
            raise TypeError("expected list of bytes")
        total_size += 4 + PyBytes_GET_SIZE(val)

    out = PyBytes_FromStringAndSize(NULL, total_size)
    start = data = <unsigned char *> PyBytes_AS_STRING(out)

    # Copy data to output.
    for i in range(n):
        val = items[i]
        # `itemlen` should be >= 0, so no signed extension issues
        itemlen = PyBytes_GET_SIZE(val)
        data[0] = itemlen & 0xff
        data[1] = (itemlen >> 8) & 0xff
        data[2] = (itemlen >> 16) & 0xff
        data[3] = (itemlen >> 24) & 0xff
        data += 4
        memcpy(data, PyBytes_AS_STRING(val), itemlen)
        data += itemlen

    assert (data - start) == total_size
    return out


def unpack_byte_array(bytes raw_bytes, Py_ssize_t n):
    """
    Unpack a variable length byte array column.
    A list of bytes objects is returned.  RuntimeError is raised
    if *raw_bytes* contents don't exactly match *n*.
    """
    cdef:
        Py_ssize_t i, itemlen, remaining
        unsigned char *start
        unsigned char *data
        list out

    start = data = <unsigned char *> PyBytes_AS_STRING(raw_bytes)
    remaining = PyBytes_GET_SIZE(raw_bytes)
    out = [None] * n

    for i in range(n):
        remaining -= 4
        # It is required to check this inside the loop to avoid
        # out of bounds array accesses.
        if remaining < 0:
            raise RuntimeError("Ran out of input")
        itemlen = (data[0] + (data[1] << 8) +
                   (data[2] << 16) + (data[3] << 24))
        data += 4

        remaining -= itemlen
        if remaining < 0:
            raise RuntimeError("Ran out of input")
        out[i] = PyBytes_FromStringAndSize(<char *> data, itemlen)
        data += itemlen

    return out

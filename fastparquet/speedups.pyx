"""
Native accelerators for Parquet encoding and decoding.
"""

from __future__ import absolute_import

import array
import numpy as np
cimport numpy as np
import cython
cimport cython
import sys
from cpython cimport (array, PyBytes_GET_SIZE, PyBytes_AS_STRING, PyBytes_Check,
                      PyBytes_FromStringAndSize, PyUnicode_AsUTF8String)
from cpython.buffer cimport (PyBUF_ANY_CONTIGUOUS, PyObject_GetBuffer,
                             PyBuffer_Release)
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free

PY2 = sys.version_info[0] == 2
_obj_dtype = np.dtype('object')

cdef extern from "Python.h":
    bytearray PyByteArray_FromStringAndSize(char *v, Py_ssize_t l)
    char* PyByteArray_AS_STRING(object string)
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    int PyUnicode_Check(object text)
    int PyBytes_AsStringAndSize(bytes obj, char **buffer, Py_ssize_t *length)
    char* PyUnicode_AsUTF8AndSize(unicode unicode, Py_ssize_t *size)
    unicode PyUnicode_DecodeUTF8(char *s, Py_ssize_t size, char *errors)


@cython.wraparound(False)
@cython.boundscheck(False)
def encode(buf, bint utf8=1):
    cdef:
        Py_ssize_t i, l, n_items, data_length, total_length
        object[:] input_values
        char** encoded_values
        int[:] encoded_lengths
        char* encv
        char* b
        bytearray out
        char* data
        object u

    # normalise input
    input_values = np.asanyarray(buf, dtype=object).reshape(-1, order='A')

    # determine number of items
    n_items = input_values.shape[0]

    # setup intermediates
    encoded_values = <char **>malloc(n_items * sizeof(char *))
    encoded_lengths = np.empty(n_items, dtype=np.intc)

    # first iteration to convert to bytes
    data_length = 0
    for i in range(n_items):
        u = input_values[i]
        if utf8:
            if not PyUnicode_Check(u):
                raise TypeError('expected unicode string, found %r' % u)
            encoded_values[i] = PyUnicode_AsUTF8AndSize(u, &l)
        else:
            PyBytes_AsStringAndSize(u, &encv, &l)
            encoded_values[i] = encv
        data_length += l + 4  # 4 bytes to store item length
        encoded_lengths[i] = l

    # setup output
    total_length = data_length
    out = PyByteArray_FromStringAndSize(NULL, total_length)

    # write header
    data = PyByteArray_AS_STRING(out)

    # second iteration, store data
    for i in range(n_items):
        l = encoded_lengths[i]
        data[0] = l & 0xff
        data[1] = (l >> 8) & 0xff
        data[2] = (l >> 16) & 0xff
        data[3] = (l >> 24) & 0xff
        data += 4
        encv = encoded_values[i]
        memcpy(data, encv, l)
        data += l

    free(encoded_values)
    return bytes(out)


@cython.wraparound(False)
@cython.boundscheck(False)
def decode(bytes buf, int n_items, bint utf8=1):
    cdef:
        char* data
        char* data_end
        Py_ssize_t i, l, input_length

    input_length = len(buf)

    # obtain input data pointer
    data = buf
    data_end = data + input_length

    # setup output
    out = np.empty(n_items, dtype=object)

    for i in range(n_items):
        if data + 4 > data_end:
            raise ValueError('corrupt buffer, data seem truncated')
        l = (data[0] + (data[1] << 8) +
             (data[2] << 16) + (data[3] << 24))
        data += 4
        if data + l > data_end:
            raise ValueError('corrupt buffer, data seem truncated')
        if utf8:
            out[i] = PyUnicode_FromStringAndSize(data, l)
        else:
            out[i] = PyBytes_FromStringAndSize(data, l)
        data += l

    return out


def array_encode(np.ndarray[object, ndim=1] data):
    cdef:
        Py_ssize_t i, n

    n = len(data)
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = PyUnicode_AsUTF8String(data[i])
    return out


def array_decode(np.ndarray data):
    cdef:
        Py_ssize_t i, n

    n = len(data)
    out = np.empty(n, dtype=object)
    for i in range(n):
        val = data[i]
        out[i] = PyUnicode_DecodeUTF8(
            PyBytes_AS_STRING(val),
            PyBytes_GET_SIZE(val),
            NULL,   # errors
            )
    return out

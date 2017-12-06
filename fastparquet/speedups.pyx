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

PY2 = sys.version_info[0] == 2
_obj_dtype = np.dtype('object')

cdef extern from "Python.h":
    bytearray PyByteArray_FromStringAndSize(char *v, Py_ssize_t l)
    char* PyByteArray_AS_STRING(object string)
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    int PyUnicode_Check(object text)


@cython.wraparound(False)
@cython.boundscheck(False)
def encodeutf8(buf):
    cdef:
        Py_ssize_t i, l, n_items, data_length, total_length
        object[:] input_values
        object[:] encoded_values
        int[:] encoded_lengths
        char* encv
        bytes b
        bytearray out
        char* data
        object u

    # normalise input
    input_values = np.asanyarray(buf, dtype=object).reshape(-1, order='A')

    # determine number of items
    n_items = input_values.shape[0]

    # setup intermediates
    encoded_values = np.empty(n_items, dtype=object)
    encoded_lengths = np.empty(n_items, dtype=np.intc)

    # first iteration to convert to bytes
    data_length = 0
    for i in range(n_items):
        u = input_values[i]
        if not PyUnicode_Check(u):
            raise TypeError('expected unicode string, found %r' % u)
        b = PyUnicode_AsUTF8String(u)
        l = PyBytes_GET_SIZE(b)
        encoded_values[i] = b
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
        encv = PyBytes_AS_STRING(encoded_values[i])
        memcpy(data, encv, l)
        data += l

    return bytes(out)


def encodebytes():
    pass


@cython.wraparound(False)
@cython.boundscheck(False)
def decodeutf8(buf, int n_items):
    cdef:
        Buffer input_buffer
        char* data
        char* data_end
        Py_ssize_t i, l, data_length, input_length

    # accept any buffer
    input_buffer = Buffer(buf, PyBUF_ANY_CONTIGUOUS)
    input_length = input_buffer.nbytes


    # obtain input data pointer
    data = input_buffer.ptr
    data_end = data + input_length

    # setup output
    out = np.empty(n_items, dtype=object)

    # iterate and decode - N.B., do not try to cast `out` as object[:]
    # as this causes segfaults, possibly similar to
    # https://github.com/cython/cython/issues/1608
    for i in range(n_items):
        if data + 4 > data_end:
            raise ValueError('corrupt buffer, data seem truncated')
        l = (data[0] + (data[1] << 8) +
             (data[2] << 16) + (data[3] << 24))
        data += 4
        if data + l > data_end:
            raise ValueError('corrupt buffer, data seem truncated')
        out[i] = PyUnicode_FromStringAndSize(data, l)
        data += l

    return out


def decodebytes():
    pass


cdef class Buffer:
    """Compatibility class to work around fact that array.array does not support
    new-style buffer interface in PY2."""
    cdef:
        char *ptr
        Py_buffer buffer
        size_t nbytes
        size_t itemsize
        array.array arr
        bint new_buffer
        bint released

    def __cinit__(self, obj, flags):
        self.released = False
        if hasattr(obj, 'dtype'):
            if obj.dtype.kind in 'Mm':
                obj = obj.view('u8')
            elif obj.dtype.kind == 'O':
                raise ValueError('cannot obtain buffer from object array')
        if PY2 and isinstance(obj, array.array):
            self.new_buffer = False
            self.arr = obj
            self.ptr = <char *> self.arr.data.as_voidptr
            self.itemsize = self.arr.itemsize
            self.nbytes = self.arr.buffer_info()[1] * self.itemsize
        else:
            self.new_buffer = True
            PyObject_GetBuffer(obj, &(self.buffer), flags)
            self.ptr = <char *> self.buffer.buf
            self.itemsize = self.buffer.itemsize
            self.nbytes = self.buffer.len

    cpdef release(self):
        if self.new_buffer and not self.released:
            PyBuffer_Release(&(self.buffer))
            self.released = True

    def __dealloc__(self):
        self.release()
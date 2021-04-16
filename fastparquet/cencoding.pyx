# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3

import cython
cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t n)


cpdef void read_rle(NumpyIO file_obj, int header, int bit_width, NumpyIO o):
    """Read a run-length encoded run from the given fo with the given header and bit_width.

    The count is determined from the header and the width is used to grab the
    value that's repeated. Yields the value repeated count times.
    """
    cdef int count, width, extra
    cdef char[:] data
    cdef char *inptr, *outptr
    count = header >> 1
    width = (bit_width + 7) // 8
    extra = 4 - width
    data = file_obj.read(width)
    inptr = file_obj.get_pointer()
    outptr = o.get_pointer()
    for _ in range(count):
        memcpy(outptr, inptr, width)
        outptr += width
        for _ in range(extra):
            outptr[0] = 0
            outptr += 1
    file_obj.seek(count * width, 1)
    o.seek(count * 4, 1)


cdef int width_from_max_int(long value):
    """Convert the value specified to a bit_width."""
    cdef int i
    for i in range(0, 64):
        if value == 0:
            return i
        value >>= 1


cdef int _mask_for_bits(int i):
    """Generate a mask to grab `i` bits from an int value."""
    return (1 << i) - 1


cpdef void read_bitpacked(NumpyIO file_obj, char header, int width, NumpyIO o):
    """
    Read values packed into width-bits each (which can be >8)
    """
    cdef unsigned int count, mask, data
    cdef unsigned char left = 8, right = 0, b = 0
    cdef int* ptr

    ptr = <int*>o.get_pointer()
    count = (header >> 1) * 8
    mask = _mask_for_bits(width)
    data = 0xff & <unsigned int>file_obj.read_byte()
    while count:
        if right > 8:
            data >>= 8
            left -= 8
            right -= 8
        elif left - right < width:
            b = file_obj.read_byte()
            data |= (<unsigned int>b & 0xff) << 8
            left += 8
        else:
            ptr[0] = <int>(data >> right & mask)
            ptr += 1
            count -= 1
            right += width
    o.seek(<char*>ptr - o.get_pointer(), 0)  # sets .loc


cpdef long read_unsigned_var_int(NumpyIO file_obj):
    """Read a value using the unsigned, variable int encoding.
    file-obj is a NumpyIO of bytes; avoids struct to allow numba-jit
    """
    cdef int result = 0, shift = 0
    cdef char byte
    while True:
        byte = file_obj.read_byte()
        result |= (<int>(byte & 0x7F) << shift)
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result


cpdef void read_rle_bit_packed_hybrid(NumpyIO io_obj, int width, int length, NumpyIO o):
    """Read values from `io_obj` using the rel/bit-packed hybrid encoding.

    If length is not specified, then a 32-bit int is read first to grab the
    length of the encoded data.

    file-obj is a NumpyIO of bytes; o if an output NumpyIO of int32

    The caller can tell the number of elements in the output by lookint
    at .tell().
    """
    cdef int start
    cdef long header
    if length is False:
        length = read_length(io_obj)
    start = io_obj.tell()
    while io_obj.tell() - start < length and o.tell() < o.nbytes:
        header = read_unsigned_var_int(io_obj)
        if header & 1 == 0:
            read_rle(io_obj, header, width, o)
        else:
            read_bitpacked(io_obj, header, width, o)
    io_obj.seek(start + length, 0)  # this should be moot


cpdef int read_length(NumpyIO file_obj):
    """ Numpy trick to get a 32-bit length from four bytes

    Equivalent to struct.unpack('<i'), but suitable for numba-jit
    """
    cdef int out
    out = (<int*> file_obj.ptr)[0]
    file_obj.seek(4, 1)
    return out


cdef class NumpyIO(object):
    """
    Read or write from a numpy arra like a file object

    This class is numba-jit-able (for specific dtypes)
    """
    cdef char[:] data
    cdef unsigned int loc, nbytes, itemsize
    cdef char* ptr

    def __init__(self, char[:] data):
        self.data = data
        self.loc = 0
        self.ptr = &data[0]
        self.nbytes = self.data.nbytes
        self.itemsize = self.data.itemsize

    cdef char* get_pointer(self):
        return self.ptr

    @property
    def len(self):
        return self.nbytes

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef char[:] read(self, int x):
        if self.loc + x > self.nbytes:
            x = self.nbytes - self.loc
        if x > 0:
            self.loc += x
        else:
            x = self.nbytes - self.loc
        return self.data[self.loc - x:self.loc]

    cpdef char read_byte(self):
        cdef char out
        out = self.ptr[self.loc]
        self.loc += 1
        return out

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void write(self, char[:] d):
        cdef int l
        l = len(d)
        self.loc += l
        self.ptr[self.loc-l:self.loc] = d

    cdef void write_byte(self, char b):
        if self.loc >= self.nbytes:
            # ignore attempt to write past end of buffer
            return
        self.ptr[self.loc] = b
        self.loc += 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void write_many(self, char b, int count):
        self.data[self.loc:self.loc+count] = b
        self.loc += count

    cpdef int tell(self):
        return self.loc

    cpdef void seek(self, int loc, int whence):
        if whence == 0:
            self.loc = loc
        elif whence == 1:
            self.loc += loc
        elif whence == 2:
            self.loc = self.nbytes + loc
        if self.loc > self.nbytes:
            self.loc = self.nbytes

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef char[:] so_far(self):
        """ In write mode, the data we have gathered until now
        """
        return self.data[:self.loc]

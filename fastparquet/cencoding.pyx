# https://cython.readthedocs.io/en/latest/src/userguide/
#   source_files_and_compilation.html#compiler-directives
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: cdivision=True
# cython: always_allow_keywords=False

import cython
cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t n)
from cpython cimport PyBytes_FromStringAndSize


cpdef void read_rle(NumpyIO file_obj, int header, int bit_width, NumpyIO o):
    """Read a run-length encoded run from the given fo with the given header and bit_width.

    The count is determined from the header and the width is used to grab the
    value that's repeated. Yields the value repeated count times.
    """
    cdef int count, width, extra, i, thedata
    cdef char *inptr, *outptr
    count = header >> 1
    width = (bit_width + 7) // 8
    extra = 4 - width
    inptr = file_obj.get_pointer()
    outptr = <char*> &thedata
    memcpy(outptr, inptr, width)
    for i in range(width, 4):
        outptr[i] = 0
    inptr = outptr
    outptr = o.get_pointer()
    for _ in range(count):
        memcpy(outptr, inptr, 4)
        outptr += 4
    file_obj.seek(width, 1)
    o.seek(count * 4, 1)


cpdef int width_from_max_int(long value):
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
    cdef unsigned int count, mask, data, offset
    cdef unsigned char left = 8, right = 0
    cdef int* ptr

    ptr = <int*>o.get_pointer()
    count = ((<int>header & 0xff) >> 1) * 8
    offset = count * 4
    mask = _mask_for_bits(width)
    data = 0xff & file_obj.read_byte()
    while count:
        if right > 8:
            data >>= 8
            left -= 8
            right -= 8
        elif left - right < width:
            data |= (file_obj.read_byte() & 0xff) << left
            left += 8
        else:
            ptr[0] = <int>(data >> right & mask)
            ptr += 1
            count -= 1
            right += width
    o.seek(offset, 1)  # sets .loc


cpdef unsigned long read_unsigned_var_int(NumpyIO file_obj):
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
    cdef unsigned long header
    if length is False:
        length = read_length(io_obj)
    start = io_obj.tell()
    while io_obj.tell() - start < length and o.tell() < o.nbytes:
        header = read_unsigned_var_int(io_obj)
        if header & 1 == 0:
            read_rle(io_obj, header, width, o)
        else:
            read_bitpacked(io_obj, header, width, o)


cpdef int read_length(NumpyIO file_obj):
    """ Numpy trick to get a 32-bit length from four bytes

    Equivalent to struct.unpack('<i'), but suitable for numba-jit
    """
    cdef int out
    out = (<int*> file_obj.ptr)[0]
    file_obj.seek(4, 1)
    return out


cdef void encode_unsigned_varint(int x, NumpyIO o):  # pragma: no cover
    while x > 127:
        o.write_byte((x & 0x7F) | 0x80)
        x >>= 7
    o.write_byte(x)


@cython.wraparound(False)
@cython.boundscheck(False)
def encode_bitpacked(int[:] values, int width, NumpyIO o):  # pragma: no cover
    """
    Write values packed into width-bits each (which can be >8)

    values is a NumbaIO array (int32)
    o is a NumbaIO output array (uint8), size=(len(values)*width)/8, rounded up.
    """

    cdef int bit_packed_count = (values.shape[0] + 7) // 8
    encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header
    cdef char right_byte_mask = 0b11111111
    cdef int bit=0, bits=0, v, counter
    for counter in range(values.shape[0]):
        v = values[counter]
        bits |= v << bit
        bit += width
        while bit >= 8:
            o.write_byte(bits & right_byte_mask)
            bit -= 8
            bits >>= 8
    if bit:
        o.write_byte(bits)



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
        return self.ptr + self.loc

    @property
    def len(self):
        return self.nbytes

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef char[:] read(self, int x):
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


@cython.wraparound(False)
@cython.boundscheck(False)
def _assemble_objects(object[:] assign, int[:] defi, int[:] rep, val, dic, d,
                      char null, null_val, int max_defi, int prev_i):
    """Dremel-assembly of arrays of values into lists

    Parameters
    ----------
    assign: array dtype O
        To insert lists into
    defi: int array
        Definition levels, max 3
    rep: int array
        Repetition levels, max 1
    dic: array of labels or None
        Applied if d is True
    d: bool
        Whether to dereference dict values
    null: bool
        Can an entry be None?
    null_val: bool
        can list elements be None
    max_defi: int
        value of definition level that corresponds to non-null
    prev_i: int
        1 + index where the last row in the previous page was inserted (0 if first page)
    """
    cdef int counter, i, re, de
    cdef int vali = 0
    cdef char started = False, have_null = False
    if d:
        # dereference dict values
        val = dic[val]
    i = prev_i
    part = []
    for counter in range(rep.shape[0]):
        de = defi[counter] if defi is not None else max_defi
        re = rep[counter]
        if not re:
            # new row - save what we have
            if started:
                assign[i] = None if have_null else part
                part = []
                i += 1
            else:
                # first time: no row to save yet, unless it's a row continued from previous page
                if vali > 0:
                    assign[i - 1].extend(part) # add the items to previous row
                    part = []
                    # don't increment i since we only filled i-1
                started = True
        if de == max_defi:
            # append real value to current item
            part.append(val[vali])
            vali += 1
        elif de > null:
            # append null to current item
            part.append(None)
        # next object is None as opposed to an object
        have_null = de == 0 and null
    if started: # normal case - add the leftovers to the next row
        assign[i] = None if have_null else part
    else: # can only happen if the only elements in this page are the continuation of the last row from previous page
        assign[i - 1].extend(part)
    return i


cdef int zigzag_int(unsigned long n):
    return (n >> 1) ^ -(n & 1)


cdef long zigzag_long(unsigned long n):
    return (n >> 1) ^ -(n & 1)


cpdef dict read_thrift(NumpyIO data):
    cdef char byte, id = 0, bit
    cdef int size
    out = {}
    while True:
        byte = data.read_byte()
        if byte == 0:
            break
        id += (byte & 0b11110000) >> 4
        bit = byte & 0b00001111
        if bit == 1:
            out[id] = True
        elif bit == 2:
            out[id] == False
        elif bit == 5 or bit == 6:
            out[id] = zigzag_long(read_unsigned_var_int(data))
        elif bit == 7:
            out[id] = <double>data.get_pointer()[0]
            data.seek(4, 1)
        elif bit == 8:
            size = read_unsigned_var_int(data)
            out[id] = PyBytes_FromStringAndSize(data.get_pointer(), size)
            data.seek(size, 1)
        elif bit == 9:
            out[id] = read_list(data)
        elif bit == 12:
            out[id] = read_thrift(data)
    return out


cdef list read_list(NumpyIO data):
    cdef char byte, typ
    cdef int size, bsize, _
    byte = data.read_byte()
    if byte >= 0xf0:  # 0b11110000
        size = read_unsigned_var_int(data)
    else:
        size = ((byte & 0xf0) >> 4)
    out = []
    typ = byte & 0x0f # 0b00001111
    if typ == 5:
        for _ in range(size):
            out.append(zigzag_int(read_unsigned_var_int(data)))
    elif typ == 8:
        for _ in range(size):
            bsize = read_unsigned_var_int(data)
            out.append(PyBytes_FromStringAndSize(data.get_pointer(), size))
            data.seek(bsize, 1)
    else:
        for _ in range(size):
            out.append(read_thrift(data))

    return out

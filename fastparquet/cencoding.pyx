# https://cython.readthedocs.io/en/latest/src/userguide/
#   source_files_and_compilation.html#compiler-directives
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
# cython: initializedcheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: always_allow_keywords=False

import cython
cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t n)
from cpython cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint32_t, int32_t, uint64_t, int64_t


cpdef void read_rle(NumpyIO file_obj, int32_t header, int32_t bit_width, NumpyIO o, int32_t itemsize=4):
    """Read a run-length encoded run from the given fo with the given header and bit_width.

    The count is determined from the header and the width is used to grab the
    value that's repeated. Yields the value repeated count times.
    """
    cdef:
        int32_t count, width, i, data = 0, vals_left
        char * inptr = file_obj.get_pointer()
        char * outptr = o.get_pointer()
    count = header >> 1
    width = (bit_width + 7) // 8
    for i in range(width):
        data |= (inptr[0] & 0xff) << (i * 8)
        inptr += 1
    vals_left = (o.nbytes - o.loc) // itemsize
    if count > vals_left:
        count = vals_left
    if itemsize == 4:
        for i in range(count):
            (<int32_t*>outptr)[0] = data
            outptr += 4
    else:
        for i in range(count):
            outptr[0] = data & 0xff
            outptr += 1
    o.loc += outptr - o.get_pointer()
    file_obj.loc += inptr - file_obj.get_pointer()


cpdef int32_t width_from_max_int(int64_t value):
    """Convert the value specified to a bit_width."""
    cdef int32_t i
    for i in range(0, 64):
        if value == 0:
            return i
        value >>= 1


cdef int32_t _mask_for_bits(int32_t i):
    """Generate a mask to grab `i` bits from an int value."""
    return (1 << i) - 1


cpdef void read_bitpacked1(NumpyIO file_obj, int32_t count, NumpyIO o):
    # implementation of np.unpackbits with output array. Output is int8 array
    cdef:
        char * inptr = file_obj.get_pointer()
        char * outptr = o.get_pointer()
        char * endptr
        unsigned char data
        int32_t counter, i, startcount=count
    if count > o.nbytes - o.loc:
        count = o.nbytes - o.loc
    for counter in range(count // 8):
        # whole bytes
        data = inptr[0]
        inptr += 1
        for i in range(8):
            outptr[0] = data & 1
            outptr += 1
            data >>= 1
    if count % 8:
        # remaining values in the last byte
        data = <int32_t>inptr[0]
        inptr += 1
        for i in range(count % 8):
            outptr[0] = data & 1
            outptr += 1
            data >>= 1
    file_obj.loc += (startcount + 7) // 8
    o.loc += count


cpdef void write_bitpacked1(NumpyIO file_obj, int32_t count, NumpyIO o):
    # implementation of np.packbits with output array. Input is int8 array
    cdef char * inptr
    cdef char * outptr
    cdef char data = 0
    cdef int32_t counter, i
    cdef int64_t indata
    outptr = o.get_pointer()
    inptr = file_obj.get_pointer()
    for counter in range(count // 8):
        # fetch a long in one op, instead of byte by byte
        indata = (<int64_t*>inptr)[0]
        inptr += 8
        for i in range(8):
            data = data << 1 | (indata & 1)
            indata >>= 8
        outptr[0] = data
        outptr += 1
    if count % 8:
        # leftover partial byte
        data = 0
        for i in range(count % 8):
            data = data << 1 | (inptr[0] != 0)
            inptr += 1
        outptr[0] = data
        outptr += 1
    file_obj.loc += count * 4
    o.loc += (count + 7) // 8


cpdef void read_bitpacked(NumpyIO file_obj, int32_t header, int32_t width, NumpyIO o, int32_t itemsize=4):
    """
    Read values packed into width-bits each (which can be >8)
    """
    cdef:
        uint32_t count, mask, data, vals_left
        unsigned char left = 8, right = 0
        char * inptr = file_obj.get_pointer()
        char * outptr = o.get_pointer()
        char * endptr

    count = ((header & 0xff) >> 1) * 8
    # TODO: special case for width=1, 2, 4, 8
    if width == 1 and itemsize == 1:
        read_bitpacked1(file_obj, count, o)
        return
    endptr = (o.nbytes - o.loc) + outptr - itemsize
    mask = _mask_for_bits(width)
    data = 0xff & <int32_t>inptr[0]
    inptr += 1
    while count:
        if right > 8:
            data >>= 8
            left -= 8
            right -= 8
        elif left - right < width:
            data |= (inptr[0] & 0xff) << left
            inptr += 1
            left += 8
        else:
            if outptr <= endptr:
                if itemsize == 4:
                    (<int32_t*>outptr)[0] = <int32_t>(data >> right & mask)
                    outptr += 4
                else:
                    outptr[0] = data >> right & mask
                    outptr += 1
            count -= 1
            right += width
    o.loc = o.loc + outptr - o.get_pointer()
    file_obj.loc += inptr - file_obj.get_pointer()


cpdef uint64_t read_unsigned_var_int(NumpyIO file_obj):
    """Read a value using the unsigned, variable int encoding.
    file-obj is a NumpyIO of bytes; avoids struct to allow numba-jit
    """
    cdef uint64_t result = 0
    cdef int32_t shift = 0
    cdef char byte
    cdef char * inptr = file_obj.get_pointer()

    while True:
        byte = inptr[0]
        inptr += 1
        result |= (<int64_t>(byte & 0x7F) << shift)
        if (byte & 0x80) == 0:
            break
        shift += 7
    file_obj.loc += inptr - file_obj.get_pointer()
    return result


cpdef void read_rle_bit_packed_hybrid(NumpyIO io_obj, int32_t width, int32_t length, NumpyIO o,
                                      int32_t itemsize=4):
    """Read values from `io_obj` using the rel/bit-packed hybrid encoding.

    If length is not specified, then a 32-bit int is read first to grab the
    length of the encoded data.

    file-obj is a NumpyIO of bytes; o if an output NumpyIO of int32 or int8/bool

    The caller can tell the number of elements in the output by looking
    at .tell().
    """
    cdef int32_t start, header
    if length is False:
        length = io_obj.read_int()
    start = io_obj.loc
    while io_obj.loc - start < length and o.loc < o.nbytes:
        header = <int32_t>read_unsigned_var_int(io_obj)
        if header & 1 == 0:
            read_rle(io_obj, header, width, o, itemsize)
        else:
            read_bitpacked(io_obj, header, width, o, itemsize)

cpdef void delta_binary_unpack(NumpyIO file_obj, NumpyIO o):
    cdef:
        uint64_t block_size = read_unsigned_var_int(file_obj)
        uint64_t miniblock_per_block = read_unsigned_var_int(file_obj)
        int64_t count = read_unsigned_var_int(file_obj)
        int32_t value = zigzag_int(read_unsigned_var_int(file_obj))
        int32_t block, min_delta, i, j, values_per_miniblock, temp
        const uint8_t[:] bitwidths
        char bitwidth, header
    values_per_miniblock = block_size // miniblock_per_block
    while True:
        min_delta = zigzag_int(read_unsigned_var_int(file_obj))
        bitwidths = file_obj.read(miniblock_per_block)
        for i in range(miniblock_per_block):
            bitwidth = bitwidths[i]
            if bitwidth:
                header = ((block_size // miniblock_per_block) // 8) << 1
                read_bitpacked(file_obj, header, bitwidth, o, itemsize=4)
                for j in range(values_per_miniblock):
                    temp = o.read_int()
                    o.seek(-4, 1)
                    o.write_int(value)
                    value += min_delta + temp
            else:
                for j in range(values_per_miniblock):
                    o.write_int(value)
                    value += min_delta
            count -= values_per_miniblock
            if count <= 0:
                return


cpdef void encode_unsigned_varint(int32_t x, NumpyIO o):  # pragma: no cover
    cdef char * outptr = o.get_pointer()
    while x > 127:
        outptr[0] = (x & 0x7F) | 0x80
        outptr += 1
        x >>= 7
    outptr[0] = x
    outptr += 1
    o.loc += outptr - o.get_pointer()


cpdef encode_bitpacked(int32_t[:] values, int32_t width, NumpyIO o):
    """
    Write values packed into width-bits each (which can be >8)
    """

    cdef int32_t bit_packed_count = (values.shape[0] + 7) // 8
    encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header
    cdef int32_t bit=0, bits=0, v, counter
    for counter in range(values.shape[0]):
        v = values[counter]
        bits |= v << bit
        bit += width
        while bit >= 8:
            o.write_byte(bits & 0xff)
            bit -= 8
            bits >>= 8
    if bit:
        o.write_byte(bits)


cpdef void encode_rle_bp(int32_t[:] data, int32_t width, NumpyIO o, int32_t withlength = 0):
    cdef uint32_t start, end
    if withlength:
        start = o.tell()
        o.seek(4, 1)
    encode_bitpacked(data, width, o)
    if withlength:
        end = o.tell()
        o.seek(start)
        o.write_int(end - start - 4)
        o.seek(end)


@cython.freelist(100)
cdef class NumpyIO(object):
    """
    Read or write from a numpy array like a file object

    The main purpose is to keep track of the current location in the memory
    """
    cdef const uint8_t[:] data
    cdef uint32_t loc, nbytes
    cdef char* ptr
    cdef char writable

    def __cinit__(self, const uint8_t[::1] data):
        self.data = data
        self.loc = 0
        self.ptr = <char*>&data[0]
        self.nbytes = data.shape[0]

    cdef char* get_pointer(self):
        return self.ptr + self.loc

    @property
    def len(self):
        return self.nbytes

    cpdef const uint8_t[:] read(self, int32_t x=-1):
        cdef const uint8_t[:] out
        if x < 1:
            x = self.nbytes - self.loc
        out = self.data[self.loc:self.loc + x]
        self.loc += x
        return out

    cpdef char read_byte(self):
        cdef char out
        out = self.ptr[self.loc]
        self.loc += 1
        return out

    cpdef int32_t read_int(self):
        cdef int32_t i
        if self.nbytes - self.loc < 4:
            return 0
        i = (<int32_t*> self.get_pointer())[0]
        self.loc += 4
        return i

    cpdef void write(self, const char[::1] d):
        memcpy(<void*>self.ptr[self.loc], <void*>&d[0], d.shape[0])
        self.loc += d.shape[0]

    cpdef void write_byte(self, char b):
        if self.loc >= self.nbytes:
            # ignore attempt to write past end of buffer
            return
        self.ptr[self.loc] = b
        self.loc += 1

    cpdef void write_int(self, int32_t i):
        if self.nbytes - self.loc < 4:
            return
        (<int32_t*> self.get_pointer())[0] = i
        self.loc += 4

    cdef void write_many(self, char b, int32_t count):
        cdef int32_t i
        for i in range(count):
            self.write_byte(b)

    cpdef int32_t tell(self):
        return self.loc

    cpdef void seek(self, int32_t loc, int32_t whence=0):
        if whence == 0:
            self.loc = loc
        elif whence == 1:
            self.loc += loc
        elif whence == 2:
            self.loc = self.nbytes + loc
        if self.loc > self.nbytes:
            self.loc = self.nbytes

    @cython.wraparound(False)
    cpdef const uint8_t[:] so_far(self):
        """ In write mode, the data we have gathered until now
        """
        return self.data[:self.loc]


def _assemble_objects(object[:] assign, const uint8_t[:] defi, const uint8_t[:] rep,
                      val, dic, d,
                      char null, null_val, int32_t max_defi, int32_t prev_i):
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
    cdef int32_t counter, i, re, de
    cdef int32_t vali = 0
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


cdef int64_t nat = -9223372036854775808


cpdef void time_shift(const int64_t[::1] data, int32_t factor=1000):
    cdef int32_t i
    cdef int64_t * ptr
    cdef int64_t value
    ptr = <int64_t*>&data[0]
    for i in range(data.shape[0]):
        if ptr[0] != nat:
            ptr[0] *= factor
        ptr += 1


cdef int32_t zigzag_int(uint64_t n):
    return (n >> 1) ^ -(n & 1)


cdef int64_t zigzag_long(uint64_t n):
    return (n >> 1) ^ -(n & 1)


cpdef dict read_thrift(NumpyIO data):
    cdef char byte, id = 0, bit
    cdef int32_t size
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
            out[id] = False
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
    cdef int32_t size, bsize, _
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

"""encoding.py - methods for reading parquet encoded data blocks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import izip

import numba


@numba.njit(nogil=True)
def encode_unsigned_varint(x, o):  # pragma: no cover
    while x > 127:
        o.write_byte((x & 0x7F) | 0x80)
        x >>= 7
    o.write_byte(x)


def remainder(value, mod):
    m = value % mod
    if m:
        return mod - m
    else:
        return 0


def ceiling(value, mod):
    m = value % mod
    if m:
        return value + mod - m
    else:
        return value


reverse_enumerate = lambda l: izip(xrange(len(l)-1, -1, -1), reversed(l))



class Encoder(object):

    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def to_bytearay(self):
        return bytearray(self.data)

    def bytes(self, data):
        self.data.extend(data)

    def byte(self, value):
        self.data.append(value)

    def zeros(self, num):
        self.data.extend([0]*num)

    def unsigned_var_int(self, value):
        while value > 127:
            self.byte((value & 0x7F) | 0x80)
            value >>= 7
        self.byte(value)  # msb marks end of value

    def fixed_int(self, value, byte_width):
        while byte_width:
            self.byte(value & 0x7F)
            value = value >> 8
            byte_width -= 1

    def set_fixed_int(self, value, byte_width, index):
        while byte_width:
            self.data[index] = (value & 0x7F)
            value >>= 8
            index += 1
            byte_width -= 1

    def bit_packed(self, values, bit_width):
        # assume there are len(values) % 8 == 0
        start = len(self)
        offset = 0
        acc = 0
        mask = _mask_for_bits(bit_width)
        for v in values:
            acc |= (v & mask) << offset
            offset += bit_width
            while offset >= 8:
                self.byte(acc & 0xFF)
                acc >> 8
                offset -= 8
        if acc:
            self.byte(acc)

        end = len(self)
        remain = remainder(end-start, bit_width)
        while remain:
            self.byte(0)
            remain -= 1

    def rle_bit_packed_hybrid(self, values, bit_width):
        self.zeros(4)
        all_start = len(self)

        rle_byte_width = (bit_width + 7) // 8
        rle_minimum_bytes = rle_byte_width + 1 + 1  # round up to closest whole byte, plus 1 for rle header, plus another for subsequent bit-packed header
        rle_threshhold = ceiling(rle_minimum_bytes * 8, bit_width) // bit_width  # the number of values we could encode in the minimum bytes (round up to be conservative)

        # find runs that are good for rle
        long_runs = []  # store (index, length) pairs
        last_value = None
        count = 1
        for i, v in reverse_enumerate(values):
            if v != last_value:
                if count > rle_threshhold:
                    long_runs.append((last_value, i + 1, count))
                count = 1
            else:
                count += 1
            last_value = v
        if count > rle_threshhold or count == len(values):
            long_runs.append((last_value, 0, count))

        index = 0
        for value, start, run_length in reversed(long_runs):
            if start > index:
                # bit packing before we do rle
                num = ceiling(start - index, 8)
                self.unsigned_var_int(num >> 2 | 1)  # (num/8)<<1 same as (num>>2) (when num is already divisible by 8)
                self.bit_packed(values[index:index + num])
                index += num

            # now we do rle
            num = run_length - index + start
            self.unsigned_var_int(num << 1)
            self.fixed_int(value, rle_byte_width)
            index += num

        # any remaining bit_packing?
        if index < len(values):
            # bit packing the rest of the values
            num = ceiling(len(values) - index, 8)
            self.unsigned_var_int(num >> 2 | 1)
            self.bit_packed(values[index:index + num], bit_width)

        self.set_fixed_int(len(self) - all_start, 4, all_start - 4)


def assemble(values, rep_levels, def_levels, schema):
    max = schema.max_definition_level()+1

    def _add(value, rep_level, def_level, parents):
        if def_level == len(parents):
            new_parents = parents[0:rep_level + 1]
            for _ in range(rep_level, max):
                new_child = []
                new_parents[-1].append(new_child)
                new_parents.append(new_child)
            new_parents[-1].append(value)
        else:
            new_parents = parents[0:def_level + 1]
            new_parents.append(None)

    rows = []
    parents = [rows]
    for value, rep_level, def_level in zip(values, def_levels, rep_levels):
        _add(parents, value, rep_level, def_level)



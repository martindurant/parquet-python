#!/usr/bin/env python

"""
AES Key Expansion.

Expands 128, 192, or 256 bit key for use with AES

Running this file as __main__ will result in a self-test of the algorithm.

Algorithm per NIST FIPS-197 http://csrc.nist.gov/publications/fips/fips197/fips-197.pdf

Copyright (c) 2010, Adam Newman http://www.caller9.com/
Licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
"""
__author__ = "Adam Newman"

# Normally use relative import. In test mode use local import.
try:
    from . import aes_tables
except ValueError:
    import aes_tables

class KeyExpander:
    """Perform AES Key Expansion"""

    _expanded_key_length = {128 : 176, 192 : 208, 256 : 240}

    def __init__(self, key_length):
        self._key_length = key_length
        self._n = int(key_length / 8)

        if key_length in self._expanded_key_length:
            self._b = self._expanded_key_length[key_length]
        else:
            raise LookupError('Invalid Key Size')

    def _core(self, key_array, iteration):
        if len(key_array) != 4:
            raise RuntimeError('_core(): key segment size invalid')

        # Append the list of elements 1-3 and list comprised of element 0 (circular rotate left)
        # For each element of this new list, put the result of sbox into output array.
        # I was torn on readability vs pythonicity. This also may be faster.
        output = [aes_tables.sbox[i] for i in key_array[1:] + key_array[:1]]

        # First byte of output array is XORed with rcon(iteration)
        output[0] = output[0] ^ aes_tables.rcon[iteration]

        return output

    def _xor_list(self, list_1, list_2):
        return [ i ^ j for i,j in zip(list_1, list_2)]

    def expand(self, key_array):
        """
            Expand the encryption key per AES key schedule specifications

            http://en.wikipedia.org/wiki/Rijndael_key_schedule# Key_schedule_description
        """

        if len(key_array) != self._n:
            raise RuntimeError('expand(): key size ' + str(len(key_array)) + ' is invalid')

        # First n bytes are copied from key. Copy prevents inplace modification of original key
        new_key = list(key_array)

        rcon_iteration = 1
        len_new_key = len(new_key)

        # There are several parts of the code below that could be done with tidy list comprehensions like
        # the one I put in _core, but I left this alone for readability.

        # Grow the key until it is the correct length
        while len_new_key < self._b:

            # Copy last 4 bytes of extended key, apply _core function order i, increment i(rcon_iteration),
            # xor with 4 bytes n bytes from end of extended key
            t = new_key[-4:]
            t = self._core(t, rcon_iteration)
            rcon_iteration += 1
            t = self._xor_list(t, new_key[-self._n : -self._n + 4])# self._n_bytes_before(len_new_key, new_key))
            new_key.extend(t)
            len_new_key += 4

            # Run three passes of 4 byte expansion using copy of 4 byte tail of extended key
            # which is then xor'd with 4 bytes n bytes from end of extended key
            for j in range(3):
                t = new_key[-4:]
                t = self._xor_list(t, new_key[-self._n : -self._n + 4])
                new_key.extend(t)
                len_new_key += 4

            # If key length is 256 and key is not complete, add 4 bytes tail of extended key
            # run through sbox before xor with 4 bytes n bytes from end of extended key
            if self._key_length == 256 and len_new_key < self._b:
                t = new_key[-4:]
                t2=[]
                for x in t:
                    t2.append(aes_tables.sbox[x])
                t = self._xor_list(t2, new_key[-self._n : -self._n + 4])
                new_key.extend(t)
                len_new_key += 4

            # If key length is 192 or 256 and key is not complete, run 2 or 3 passes respectively
            # of 4 byte tail of extended key xor with 4 bytes n bytes from end of extended key
            if self._key_length != 128 and len_new_key < self._b:
                if self._key_length == 192:
                    r = range(2)
                else:
                    r = range(3)

                for j in r:
                    t = new_key[-4:]
                    t = self._xor_list(t, new_key[-self._n : -self._n + 4])
                    new_key.extend(t)
                    len_new_key += 4

        return new_key

import unittest
class TestKeyExpander(unittest.TestCase):

    def test_keys(self):
        """Test All Key Expansions"""
        try:
            from . import test_keys
        except:
            import test_keys

        test_data = test_keys.TestKeys()

        for key_size in [128, 192, 256]:
            test_expander = KeyExpander(key_size)
            test_expanded_key = test_expander.expand(test_data.test_key[key_size])
            self.assertEqual (len([i for i, j in zip(test_expanded_key, test_data.test_expanded_key_validated[key_size]) if i == j]),
                len(test_data.test_expanded_key_validated[key_size]),
                msg='Key expansion ' + str(key_size) + ' bit')

if __name__ == "__main__":
    unittest.main()




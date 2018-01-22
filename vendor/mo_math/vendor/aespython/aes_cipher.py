#!/usr/bin/env python
"""
AES Block Cipher.

Performs single block cipher decipher operations on a 16 element list of integers.
These integers represent 8 bit bytes in a 128 bit block.
The result of cipher or decipher operations is the transformed 16 element list of integers.

Running this file as __main__ will result in a self-test of the algorithm.

Algorithm per NIST FIPS-197 http://csrc.nist.gov/publications/fips/fips197/fips-197.pdf

Thanks to serprex for many optimizations in this code. For even more, see his github fork of this project.

Copyright (c) 2010, Adam Newman http://www.caller9.com/
                    Demur Rumed https://github.com/serprex
Licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
"""
from mo_logs import Log

__author__ = "Adam Newman"

# Normally use relative import. In test mode use local import.
try:
    from . import aes_tables
except ValueError:
    import aes_tables

class AESCipher:
    """Perform single block AES cipher/decipher"""

    def __init__ (self, expanded_key):
        # Store epanded key
        self._expanded_key = expanded_key

        # Number of rounds determined by expanded key length
        self._Nr = int(len(expanded_key) / 16) - 1

    def _sub_bytes (self, state):
        # Run state through sbox
        for i,s in enumerate(state):state[i]=aes_tables.sbox[s]

    def _i_sub_bytes (self, state):
        # Run state through inverted sbox
        for i,s in enumerate(state):state[i]=aes_tables.i_sbox[s]

    def _shift_row (self, row, shift):
        # Circular shift row left by shift amount
        row+=row[:shift]
        del row[:shift]
        return row

    def _i_shift_row (self, row, shift):
        # Circular shift row left by shift amount
        row+=row[:shift]
        del row[:4+shift]
        return row

    def _shift_rows (self, state):
        # Extract rows as every 4th item starting at [1..3]
        # Replace row with shift_row operation
        for i in 1,2,3:
            state[i::4] = self._shift_row(state[i::4],i)

    def _i_shift_rows (self, state):
        # Extract rows as every 4th item starting at [1..3]
        # Replace row with inverse shift_row operation
        for i in 1,2,3:
            state[i::4] = self._i_shift_row(state[i::4],-i)

    def _mix_column (self, column, inverse):
        # Use galois lookup tables instead of performing complicated operations
        # If inverse, use matrix with inverse values
        g0,g1,g2,g3=aes_tables.galI if inverse else aes_tables.galNI
        c0,c1,c2,c3=column
        return (
            g0[c0]^g1[c1]^g2[c2]^g3[c3],
            g3[c0]^g0[c1]^g1[c2]^g2[c3],
            g2[c0]^g3[c1]^g0[c2]^g1[c3],
            g1[c0]^g2[c1]^g3[c2]^g0[c3])

    def _mix_columns (self, state, inverse):
        # Perform mix_column for each column in the state
        for i,j in (0,4),(4,8),(8,12),(12,16):
            state[i:j] = self._mix_column(state[i:j], inverse)

    def _add_round_key (self, state, round):
        # XOR the state with the current round key
        for k, (i, j) in enumerate(zip(state, self._expanded_key[round * 16:(round + 1) * 16])):
            state[k] = i ^ j

    def cipher_block (self, state):
        """Perform AES block cipher on input"""
        # PKCS7 Padding
        state=state+[16-len(state)]*(16-len(state))# Fails test if it changes the input with +=

        self._add_round_key(state, 0)

        for i in range(1, self._Nr):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state, False)
            self._add_round_key(state, i)

        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self._Nr)
        return state

    def decipher_block (self, state):
        """Perform AES block decipher on input"""
        if len(state) != 16:
            Log.error(u"Expecting block of 16")

        self._add_round_key(state, self._Nr)

        for i in range(self._Nr - 1, 0, -1):
            self._i_shift_rows(state)
            self._i_sub_bytes(state)
            self._add_round_key(state, i)
            self._mix_columns(state, True)

        self._i_shift_rows(state)
        self._i_sub_bytes(state)
        self._add_round_key(state, 0)
        return state

import unittest
class TestCipher(unittest.TestCase):
    def test_cipher(self):
        """Test AES cipher with all key lengths"""
        try:
            from . import test_keys, key_expander
        except:
            import test_keys, key_expander

        test_data = test_keys.TestKeys()

        for key_size in 128, 192, 256:
            test_key_expander = key_expander.KeyExpander(key_size)
            test_expanded_key = test_key_expander.expand(test_data.test_key[key_size])
            test_cipher = AESCipher(test_expanded_key)
            test_result_ciphertext = test_cipher.cipher_block(test_data.test_block_plaintext)
            self.assertEquals(len([i for i, j in zip(test_result_ciphertext, test_data.test_block_ciphertext_validated[key_size]) if i == j]),
                16,
                msg='Test %d bit cipher'%key_size)

            test_result_plaintext = test_cipher.decipher_block(test_data.test_block_ciphertext_validated[key_size])
            self.assertEquals(len([i for i, j in zip(test_result_plaintext, test_data.test_block_plaintext) if i == j]),
                16,
                msg='Test %d bit decipher'%key_size)

if __name__ == "__main__":
    unittest.main()

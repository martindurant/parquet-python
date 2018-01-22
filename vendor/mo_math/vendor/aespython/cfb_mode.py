#!/usr/bin/env python
"""
CFB Mode of operation

Running this file as __main__ will result in a self-test of the algorithm.

Algorithm per NIST SP 800-38A http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

Copyright (c) 2010, Adam Newman http://www.caller9.com/
Licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
"""
__author__ = "Adam Newman"

from .cipher_mode import CipherMode
from .mode_test import GeneralTestEncryptionMode

class CFBMode(CipherMode):
    """Perform CFB operation on a block and retain IV information for next operation"""

    name = "CFB"

    def __init__(self, block_cipher, block_size):
        CipherMode.__init__(self, block_cipher, block_size)

    def encrypt_block(self, plaintext):
        cipher_iv = self._block_cipher.cipher_block(self._iv)
        ciphertext = [i ^ j for i,j in zip (plaintext, cipher_iv)]
        self._iv = ciphertext
        return ciphertext

    def decrypt_block(self, ciphertext):
        cipher_iv = self._block_cipher.cipher_block(self._iv)
        plaintext = [i ^ j for i,j in zip (cipher_iv, ciphertext)]
        self._iv = ciphertext
        return plaintext

class TestEncryptionMode(GeneralTestEncryptionMode):
    def test_mode(self):
        """Test CBC Mode Encrypt/Decrypt"""
        try:
            from .test_keys import TestKeys
        except:
            from test_keys import TestKeys

        test_data = TestKeys()

        test_mode = CFBMode(self.get_keyed_cipher(test_data.test_mode_key), 16)

        self.run_cipher(test_mode, test_data.test_mode_iv, test_data.test_cfb_ciphertext, test_data.test_mode_plaintext)

if __name__ == "__main__":
    import unittest
    unittest.main()

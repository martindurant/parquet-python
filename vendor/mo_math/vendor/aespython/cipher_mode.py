#!/usr/bin/env python
"""
Cipher Mode of operation

Running this file as __main__ will result in a self-test of the algorithm.

Algorithm per NIST SP 800-38A http://csrc.nist.gov/publications/nistpubs/800-38a/sp800-38a.pdf

Copyright (c) 2010, Adam Newman http://www.caller9.com/
Licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
"""
from mo_logs import Log

__author__ = "Adam Newman"

class CipherMode:
    """Perform Cipher operation on a block and retain IV information for next operation"""

    name = "ABSTRACT"

    def __init__(self, block_cipher, block_size):
        self._block_cipher = block_cipher
        self._block_size = block_size
        self._iv = [0] * block_size

    def set_iv(self, iv):
        if len(iv) != self._block_size:
            Log.error("iv is wrong size")
        self._iv = iv

    def encrypt_block(self, plaintext):
        raise(NotImplementedError, "Abstract function")

    def decrypt_block(self, ciphertext):
        raise(NotImplementedError, "Abstract function")



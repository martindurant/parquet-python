"""
Cipher Mode of operation

Abstract encryption mode test harness.
"""
from .key_expander import KeyExpander
from .aes_cipher import AESCipher

import unittest
class GeneralTestEncryptionMode(unittest.TestCase):
    def get_keyed_cipher(self, key):


        test_expander = KeyExpander(256)
        test_expanded_key = test_expander.expand(key)

        return AESCipher(test_expanded_key)

    def run_cipher(self, cipher_mode, iv, ciphertext_list, plaintext_list):
        """Given an cipher mode, test key, and test iv, use known ciphertext, plaintext to test algorithm"""

        cipher_mode.set_iv(iv)
        for k in range(len(ciphertext_list)):
            self.assertEquals(len([i for i, j in zip(ciphertext_list[k],cipher_mode.encrypt_block(plaintext_list[k])) if i == j]),
                16,
                msg=cipher_mode.name + ' encrypt test block' + str(k))

        cipher_mode.set_iv(iv)
        for k in range(len(plaintext_list)):
            self.assertEquals(len([i for i, j in zip(plaintext_list[k],cipher_mode.decrypt_block(ciphertext_list[k])) if i == j]),
                16,
                msg=cipher_mode.name + ' decrypt test block' + str(k))

    def test_mode(self):
        """Abstract Test Harness for Encrypt/Decrypt"""
        pass


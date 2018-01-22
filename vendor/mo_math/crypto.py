# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import base64

from mo_future import text_type, binary_type
from mo_dots import Data, get_module
from mo_logs import Log

from mo_math.randoms import Random
from mo_math.vendor.aespython import key_expander, aes_cipher, cbc_mode

DEBUG = False


def encrypt(text, _key, salt=None):
    """
    RETURN JSON OF ENCRYPTED DATA   {"salt":s, "length":l, "data":d}
    """
    if not isinstance(text, text_type):
        Log.error("only unicode is encrypted")
    if _key is None:
        Log.error("Expecting a key")
    if isinstance(_key, str):
        _key = bytearray(_key)
    if salt is None:
        salt = Random.bytes(16)

    data = bytearray(text.encode("utf8"))

    # Initialize encryption using key and iv
    key_expander_256 = key_expander.KeyExpander(256)
    expanded_key = key_expander_256.expand(_key)
    aes_cipher_256 = aes_cipher.AESCipher(expanded_key)
    aes_cbc_256 = cbc_mode.CBCMode(aes_cipher_256, 16)
    aes_cbc_256.set_iv(salt)

    output = Data()
    output.type = "AES256"
    output.salt = bytes2base64(salt)
    output.length = len(data)

    encrypted = bytearray()
    for _, d in _groupby16(data):
        encrypted.extend(aes_cbc_256.encrypt_block(d))
    output.data = bytes2base64(encrypted)
    json = get_module("mo_json").value2json(output)

    if DEBUG:
        test = decrypt(json, _key)
        if test != text:
            Log.error("problem with encryption")

    return json


def decrypt(data, _key):
    """
    ACCEPT JSON OF ENCRYPTED DATA  {"salt":s, "length":l, "data":d}
    """
    # Key and iv have not been generated or provided, bail out
    if _key is None:
        Log.error("Expecting a key")

    _input = get_module("mo_json").json2value(data)

    # Initialize encryption using key and iv
    key_expander_256 = key_expander.KeyExpander(256)
    expanded_key = key_expander_256.expand(_key)
    aes_cipher_256 = aes_cipher.AESCipher(expanded_key)
    aes_cbc_256 = cbc_mode.CBCMode(aes_cipher_256, 16)
    aes_cbc_256.set_iv(base642bytearray(_input.salt))

    raw = base642bytearray(_input.data)
    out_data = bytearray()
    for _, e in _groupby16(raw):
        out_data.extend(aes_cbc_256.decrypt_block(e))

    return binary_type(out_data[:_input.length:]).decode("utf8")


def bytes2base64(value):
    if isinstance(value, bytearray):
        value = binary_type(value)
    return base64.b64encode(value).decode("utf8")


def base642bytearray(value):
    if value == None:
        return bytearray(b"")
    else:
        return bytearray(base64.b64decode(value))


def _groupby16(bytes):
    count = 0
    index = 0
    length = len(bytes)
    while index < length:
        yield count, bytes[index: index + 16]
        count += 1
        index += 16

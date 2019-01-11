# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
import base64

from mo_dots import Data, get_module
from mo_future import PY2, binary_type, text_type
from mo_logs import Log
from mo_math.randoms import Random
from mo_math.vendor.aespython import aes_cipher, cbc_mode, key_expander

DEBUG = False


def encrypt(text, _key, salt=None):
    """
    RETURN {"salt":s, "length":l, "data":d} -> JSON -> UTF8
    """

    if is_text(text):
        encoding = 'utf8'
        data = bytearray(text.encode("utf8"))
    elif is_binary(text):
        encoding = None
        if PY2:
            data = bytearray(text)
        else:
            data = text

    if _key is None:
        Log.error("Expecting a key")
    if is_binary(_key):
        _key = bytearray(_key)
    if salt is None:
        salt = Random.bytes(16)

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
    output.encoding = encoding

    encrypted = bytearray()
    for _, d in _groupby16(data):
        encrypted.extend(aes_cbc_256.encrypt_block(d))
    output.data = bytes2base64(encrypted)
    json = get_module("mo_json").value2json(output, pretty=True).encode('utf8')

    if DEBUG:
        test = decrypt(json, _key)
        if test != text:
            Log.error("problem with encryption")

    return json


def decrypt(data, _key):
    """
    ACCEPT BYTES -> UTF8 -> JSON -> {"salt":s, "length":l, "data":d}
    """
    # Key and iv have not been generated or provided, bail out
    if _key is None:
        Log.error("Expecting a key")

    _input = get_module("mo_json").json2value(data.decode('utf8'), leaves=False, flexible=False)

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

    if _input.encoding:
        return binary_type(out_data[:_input.length:]).decode(_input.encoding)
    else:
        return binary_type(out_data[:_input.length:])



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

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

from collections import Mapping

from mo_dots import split_field, join_field

TYPE_PREFIX = "~"   # u'\u0442\u0443\u0440\u0435-'  # "туре"
BOOLEAN_TYPE = TYPE_PREFIX+"b~"
NUMBER_TYPE = TYPE_PREFIX+"n~"
STRING_TYPE = TYPE_PREFIX+"s~"
NESTED_TYPE = TYPE_PREFIX+"N~"
EXISTS_TYPE = TYPE_PREFIX+"e~"


def encode_property(name):
    return name.replace(",", "\\,").replace(".", ",")


def decode_property(encoded):
    return encoded.replace("\\,", "\a").replace(",", ".").replace("\a", ",")


def untype_path(encoded):
    if encoded.startswith(".."):
        remainder = encoded.lstrip(".")
        back = len(encoded) - len(remainder) - 1
        return ("." * back) + join_field(decode_property(c) for c in split_field(remainder) if not c.startswith(TYPE_PREFIX))
    else:
        return join_field(decode_property(c) for c in split_field(encoded) if not c.startswith(TYPE_PREFIX))


def unnest_path(encoded):
    if encoded.startswith(".."):
        encoded = encoded.lstrip(".")
        if not encoded:
            encoded = "."

    return join_field(decode_property(c) for c in split_field(encoded) if c != NESTED_TYPE)


def untyped(value):
    return _untype(value)


def _untype(value):
    if isinstance(value, Mapping):
        output = {}

        for k, v in value.items():
            if k == EXISTS_TYPE:
                continue
            elif k.startswith(TYPE_PREFIX):
                return v
            else:
                output[decode_property(k)] = _untype(v)
        return output
    elif isinstance(value, list):
        return [_untype(v) for v in value]
    else:
        return value

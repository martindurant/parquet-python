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

import json
import math
import sys
import time
from collections import Mapping
from datetime import datetime, date, timedelta
from decimal import Decimal
from json.encoder import encode_basestring
from math import floor

from mo_dots import Data, FlatList, NullType, Null
from mo_future import text_type, binary_type, long, utf8_json_encoder, sort_using_key, xrange
from mo_json import ESCAPE_DCT, scrub, float2json
from mo_logs import Except
from mo_logs.strings import utf82unicode, quote
from mo_times.dates import Date
from mo_times.durations import Duration

json_decoder = json.JSONDecoder().decode
_get = object.__getattribute__

_ = Except

# THIS FILE EXISTS TO SERVE AS A FAST REPLACEMENT FOR JSON ENCODING
# THE DEFAULT JSON ENCODERS CAN NOT HANDLE A DIVERSITY OF TYPES *AND* BE FAST
#
# 1) WHEN USING cPython, WE HAVE NO COMPILER OPTIMIZATIONS: THE BEST STRATEGY IS TO
# CONVERT THE MEMORY STRUCTURE TO STANDARD TYPES AND SEND TO THE INSANELY FAST
#    DEFAULT JSON ENCODER
# 2) WHEN USING PYPY, WE USE CLEAR-AND-SIMPLE PROGRAMMING SO THE OPTIMIZER CAN DO
#    ITS JOB.  ALONG WITH THE UnicodeBuilder WE GET NEAR C SPEEDS

use_pypy = False

COMMA = u","
QUOTE = u'"'
COLON = u":"
QUOTE_COLON = QUOTE + COLON
COMMA_QUOTE = COMMA + QUOTE

PRETTY_COMMA = u", "
PRETTY_COLON = u": "

try:
    # UnicodeBuilder IS ABOUT 2x FASTER THAN list()
    from __pypy__.builders import UnicodeBuilder

    use_pypy = True
except Exception as e:
    if use_pypy:
        sys.stdout.write(
            b"*********************************************************\n"
            b"** The PyLibrary JSON serializer for PyPy is in use!\n"
            b"** Currently running CPython: This will run sloooow!\n"
            b"*********************************************************\n"
        )

    class UnicodeBuilder(list):
        def __init__(self, length=None):
            list.__init__(self)

        def build(self):
            return u"".join(self)

append = UnicodeBuilder.append

_dealing_with_problem = False


def pypy_json_encode(value, pretty=False):
    """
    pypy DOES NOT OPTIMIZE GENERATOR CODE WELL
    """
    global _dealing_with_problem
    if pretty:
        return pretty_json(value)

    try:
        _buffer = UnicodeBuilder(2048)
        _value2json(value, _buffer)
        output = _buffer.build()
        return output
    except Exception as e:
        # THE PRETTY JSON WILL PROVIDE MORE DETAIL ABOUT THE SERIALIZATION CONCERNS
        from mo_logs import Log

        if _dealing_with_problem:
            Log.error("Serialization of JSON problems", e)
        else:
            Log.warning("Serialization of JSON problems", e)
        _dealing_with_problem = True
        try:
            return pretty_json(value)
        except Exception as f:
            Log.error("problem serializing object", f)
        finally:
            _dealing_with_problem = False


class cPythonJSONEncoder(object):
    def __init__(self, sort_keys=True):
        object.__init__(self)

        self.encoder = utf8_json_encoder

    def encode(self, value, pretty=False):
        if pretty:
            return pretty_json(value)

        try:
            scrubbed = scrub(value)
            return text_type(self.encoder(scrubbed))
        except Exception as e:
            from mo_logs.exceptions import Except
            from mo_logs import Log

            e = Except.wrap(e)
            Log.warning("problem serializing {{type}}", type=text_type(repr(value)), cause=e)
            raise e


def ujson_encode(value, pretty=False):
    if pretty:
        return pretty_json(value)

    try:
        scrubbed = scrub(value)
        return ujson_dumps(scrubbed, ensure_ascii=False, sort_keys=True, escape_forward_slashes=False).decode('utf8')
    except Exception as e:
        from mo_logs.exceptions import Except
        from mo_logs import Log

        e = Except.wrap(e)
        Log.warning("problem serializing {{type}}", type=text_type(repr(value)), cause=e)
        raise e


def _value2json(value, _buffer):
    try:
        _class = value.__class__
        if value is None:
            append(_buffer, u"null")
            return
        elif value is True:
            append(_buffer, u"true")
            return
        elif value is False:
            append(_buffer, u"false")
            return

        type = value.__class__
        if type is binary_type:
            append(_buffer, QUOTE)
            try:
                v = utf82unicode(value)
            except Exception as e:
                problem_serializing(value, e)

            for c in v:
                append(_buffer, ESCAPE_DCT.get(c, c))
            append(_buffer, QUOTE)
        elif type is text_type:
            append(_buffer, QUOTE)
            for c in value:
                append(_buffer, ESCAPE_DCT.get(c, c))
            append(_buffer, QUOTE)
        elif type is dict:
            if not value:
                append(_buffer, u"{}")
            else:
                _dict2json(value, _buffer)
            return
        elif type is Data:
            d = _get(value, "_dict")  # MIGHT BE A VALUE NOT A DICT
            _value2json(d, _buffer)
            return
        elif type in (int, long, Decimal):
            append(_buffer, float2json(value))
        elif type is float:
            if math.isnan(value) or math.isinf(value):
                append(_buffer, u'null')
            else:
                append(_buffer, float2json(value))
        elif type in (set, list, tuple, FlatList):
            _list2json(value, _buffer)
        elif type is date:
            append(_buffer, float2json(time.mktime(value.timetuple())))
        elif type is datetime:
            append(_buffer, float2json(time.mktime(value.timetuple())))
        elif type is Date:
            append(_buffer, float2json(value.unix))
        elif type is timedelta:
            append(_buffer, float2json(value.total_seconds()))
        elif type is Duration:
            append(_buffer, float2json(value.seconds))
        elif type is NullType:
            append(_buffer, u"null")
        elif isinstance(value, Mapping):
            if not value:
                append(_buffer, u"{}")
            else:
                _dict2json(value, _buffer)
            return
        elif hasattr(value, '__data__'):
            d = value.__data__()
            _value2json(d, _buffer)
        elif hasattr(value, '__json__'):
            j = value.__json__()
            append(_buffer, j)
        elif hasattr(value, '__iter__'):
            _iter2json(value, _buffer)
        else:
            from mo_logs import Log

            Log.error(text_type(repr(value)) + " is not JSON serializable")
    except Exception as e:
        from mo_logs import Log

        Log.error(text_type(repr(value)) + " is not JSON serializable", cause=e)


def _list2json(value, _buffer):
    if not value:
        append(_buffer, u"[]")
    else:
        sep = u"["
        for v in value:
            append(_buffer, sep)
            sep = COMMA
            _value2json(v, _buffer)
        append(_buffer, u"]")


def _iter2json(value, _buffer):
    append(_buffer, u"[")
    sep = u""
    for v in value:
        append(_buffer, sep)
        sep = COMMA
        _value2json(v, _buffer)
    append(_buffer, u"]")


def _dict2json(value, _buffer):
    try:
        prefix = u"{\""
        for k, v in value.items():
            append(_buffer, prefix)
            prefix = COMMA_QUOTE
            if isinstance(k, binary_type):
                k = utf82unicode(k)
            for c in k:
                append(_buffer, ESCAPE_DCT.get(c, c))
            append(_buffer, QUOTE_COLON)
            _value2json(v, _buffer)
        append(_buffer, u"}")
    except Exception as e:
        from mo_logs import Log

        Log.error(text_type(repr(value)) + " is not JSON serializable", cause=e)

ARRAY_ROW_LENGTH = 80
ARRAY_ITEM_MAX_LENGTH = 30
ARRAY_MAX_COLUMNS = 10
INDENT = "    "


def pretty_json(value):
    try:
        if value is False:
            return "false"
        elif value is True:
            return "true"
        elif isinstance(value, Mapping):
            try:
                items = sort_using_key(list(value.items()), lambda r: r[0])
                values = [encode_basestring(k) + PRETTY_COLON + indent(pretty_json(v)).strip() for k, v in items if v != None]
                if not values:
                    return "{}"
                elif len(values) == 1:
                    return "{" + values[0] + "}"
                else:
                    return "{\n" + INDENT + (",\n" + INDENT).join(values) + "\n}"
            except Exception as e:
                from mo_logs import Log
                from mo_math import OR

                if OR(not isinstance(k, text_type) for k in value.keys()):
                    Log.error(
                        "JSON must have string keys: {{keys}}:",
                        keys=[k for k in value.keys()],
                        cause=e
                    )

                Log.error(
                    "problem making dict pretty: keys={{keys}}:",
                    keys=[k for k in value.keys()],
                    cause=e
                )
        elif value in (None, Null):
            return "null"
        elif isinstance(value, (text_type, binary_type)):
            if isinstance(value, binary_type):
                value = utf82unicode(value)
            try:
                return quote(value)
            except Exception as e:
                from mo_logs import Log

                try:
                    Log.note("try explicit convert of string with length {{length}}", length=len(value))
                    acc = [QUOTE]
                    for c in value:
                        try:
                            try:
                                c2 = ESCAPE_DCT[c]
                            except Exception:
                                c2 = c
                            c3 = text_type(c2)
                            acc.append(c3)
                        except BaseException:
                            pass
                            # Log.warning("odd character {{ord}} found in string.  Ignored.",  ord= ord(c)}, cause=g)
                    acc.append(QUOTE)
                    output = u"".join(acc)
                    Log.note("return value of length {{length}}", length=len(output))
                    return output
                except BaseException as f:
                    Log.warning("can not even explicit convert {{type}}", type=f.__class__.__name__, cause=f)
                    return "null"
        elif isinstance(value, list):
            if not value:
                return "[]"

            if ARRAY_MAX_COLUMNS == 1:
                return "[\n" + ",\n".join([indent(pretty_json(v)) for v in value]) + "\n]"

            if len(value) == 1:
                j = pretty_json(value[0])
                if j.find("\n") >= 0:
                    return "[\n" + indent(j) + "\n]"
                else:
                    return "[" + j + "]"

            js = [pretty_json(v) for v in value]
            max_len = max(*[len(j) for j in js])
            if max_len <= ARRAY_ITEM_MAX_LENGTH and max(*[j.find("\n") for j in js]) == -1:
                # ALL TINY VALUES
                num_columns = max(1, min(ARRAY_MAX_COLUMNS, int(floor((ARRAY_ROW_LENGTH + 2.0) / float(max_len + 2)))))  # +2 TO COMPENSATE FOR COMMAS
                if len(js) <= num_columns:  # DO NOT ADD \n IF ONLY ONE ROW
                    return "[" + PRETTY_COMMA.join(js) + "]"
                if num_columns == 1:  # DO NOT rjust IF THERE IS ONLY ONE COLUMN
                    return "[\n" + ",\n".join([indent(pretty_json(v)) for v in value]) + "\n]"

                content = ",\n".join(
                    PRETTY_COMMA.join(j.rjust(max_len) for j in js[r:r + num_columns])
                    for r in xrange(0, len(js), num_columns)
                )
                return "[\n" + indent(content) + "\n]"

            pretty_list = js

            output = ["[\n"]
            for i, p in enumerate(pretty_list):
                try:
                    if i > 0:
                        output.append(",\n")
                    output.append(indent(p))
                except Exception:
                    from mo_logs import Log

                    Log.warning("problem concatenating string of length {{len1}} and {{len2}}",
                        len1=len("".join(output)),
                        len2=len(p)
                    )
            output.append("\n]")
            try:
                return "".join(output)
            except Exception as e:
                from mo_logs import Log

                Log.error("not expected", cause=e)
        elif hasattr(value, '__data__'):
            d = value.__data__()
            return pretty_json(d)
        elif hasattr(value, '__json__'):
            j = value.__json__()
            if j == None:
                return "   null   "  # TODO: FIND OUT WHAT CAUSES THIS
            return pretty_json(json_decoder(j))
        elif scrub(value) is None:
            return "null"
        elif hasattr(value, '__iter__'):
            return pretty_json(list(value))
        elif hasattr(value, '__call__'):
            return "null"
        else:
            try:
                if int(value) == value:
                    return text_type(int(value))
            except Exception:
                pass

            try:
                if float(value) == value:
                    return text_type(float(value))
            except Exception:
                pass

            return pypy_json_encode(value)

    except Exception as e:
        problem_serializing(value, e)


def problem_serializing(value, e=None):
    """
    THROW ERROR ABOUT SERIALIZING
    """
    from mo_logs import Log

    try:
        typename = type(value).__name__
    except Exception:
        typename = "<error getting name>"

    try:
        rep = text_type(repr(value))
    except Exception as _:
        rep = None

    if rep == None:
        Log.error(
            "Problem turning value of type {{type}} to json",
            type=typename,
            cause=e
        )
    else:
        Log.error(
            "Problem turning value ({{value}}) of type {{type}} to json",
            value=rep,
            type=typename,
            cause=e
        )


def indent(value, prefix=INDENT):
    try:
        content = value.rstrip()
        suffix = value[len(content):]
        lines = content.splitlines()
        return prefix + (u"\n" + prefix).join(lines) + suffix
    except Exception as e:
        raise Exception(u"Problem with indent of value (" + e.message + u")\n" + value)


def value_compare(a, b):
    if a == None:
        if b == None:
            return 0
        return -1
    elif b == None:
        return 1

    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0


def datetime2milli(d, type):
    try:
        if type == datetime:
            diff = d - datetime(1970, 1, 1)
        else:
            diff = d - date(1970, 1, 1)

        return long(diff.total_seconds()) * long(1000) + long(diff.microseconds / 1000)
    except Exception as e:
        problem_serializing(d, e)


def unicode_key(key):
    """
    CONVERT PROPERTY VALUE TO QUOTED NAME OF SAME
    """
    if not isinstance(key, (text_type, binary_type)):
        from mo_logs import Log
        Log.error("{{key|quote}} is not a valid key", key=key)
    return quote(text_type(key))


# OH HUM, cPython with uJSON, OR pypy WITH BUILTIN JSON?
# http://liangnuren.wordpress.com/2012/08/13/python-json-performance/
# http://morepypy.blogspot.ca/2011/10/speeding-up-json-encoding-in-pypy.html
if use_pypy:
    json_encoder = pypy_json_encode
else:
    # from ujson import dumps as ujson_dumps
    # json_encoder = ujson_encode
    json_encoder = cPythonJSONEncoder().encode



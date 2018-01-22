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
# REMOVED SO LOGIC SWITCHES FROM BYTES TO STRING BETWEEN PY2 AND PY3 RESPECTIVELY
# from __future__ import unicode_literals

from collections import Mapping

from mo_dots import wrap, Data
from mo_future import text_type, PY3
from mo_future import urlparse

_value2json = None
_json2value = None
_Log = None


def _late_import():
    global _value2json
    global _json2value
    global _Log

    from mo_json import value2json as value2json_
    from mo_json import json2value as json2value_
    from mo_logs import Log as _Log

    if PY3:
        _value2json = value2json_
        _json2value = json2value_
    else:
        _value2json = lambda v: value2json_(v).encode('latin1')
        _json2value = lambda v: json2value_(v.decode('latin1'))

    _ = _Log


class URL(object):
    """
    JUST LIKE urllib.parse() [1], BUT CAN HANDLE JSON query PARAMETERS

    [1] https://docs.python.org/3/library/urllib.parse.html
    """

    def __init__(self, value):
        try:
            self.scheme = None
            self.host = None
            self.port = None
            self.path = ""
            self.query = ""
            self.fragment = ""

            if value == None:
                return

            if value.startswith("file://") or value.startswith("//"):
                # urlparse DOES NOT WORK IN THESE CASES
                scheme, suffix = value.split("//", 1)
                self.scheme = scheme.rstrip(":")
                parse(self, suffix, 0, 1)
                self.query = wrap(url_param2value(self.query))
            else:
                output = urlparse(value)
                self.scheme = output.scheme
                self.port = output.port
                self.host = output.netloc.split(":")[0]
                self.path = output.path
                self.query = wrap(url_param2value(output.query))
                self.fragment = output.fragment
        except Exception as e:
            if not _Log:
                _late_import()

            _Log.error(u"problem parsing {{value}} to URL", value=value, cause=e)

    def __nonzero__(self):
        if self.scheme or self.host or self.port or self.path or self.query or self.fragment:
            return True
        return False

    def __bool__(self):
        if self.scheme or self.host or self.port or self.path or self.query or self.fragment:
            return True
        return False

    def __unicode__(self):
        return self.__str__().decode('utf8')  # ASSUME chr<128 ARE VALID UNICODE

    def __str__(self):
        url = ""
        if self.host:
            url = self.host
        if self.scheme:
            url = self.scheme + "://" + url
        if self.port:
            url = url + ":" + str(self.port)
        if self.path:
            if self.path[0] == "/":
                url += str(self.path)
            else:
                url += "/" + str(self.path)
        if self.query:
            url = url + "?" + value2url_param(self.query)
        if self.fragment:
            url = url + "#" + value2url_param(self.fragment)
        return url


def int_to_hex(value, size):
    return (("0" * size) + hex(value)[2:])[-size:]

_str_to_url = {chr(i): chr(i) for i in range(32, 128)}
for c in " {}<>;/?:@&=+$,":
    _str_to_url[c] = "%" + int_to_hex(ord(c), 2)
for i in range(128, 256):
    _str_to_url[chr(i)] = "%" + int_to_hex(i, 2)

_url_to_str = {v: k for k, v in _str_to_url.items()}

names = ["path", "query", "fragment"]
indicator = ["/", "?", "#"]


def parse(output, suffix, curr, next):
    if next == len(indicator):
        output.__setattr__(names[curr], suffix)
        return

    e = suffix.find(indicator[next])
    if e == -1:
        parse(output, suffix, curr, next + 1)
    else:
        output.__setattr__(names[curr], suffix[:e:])
        parse(output, suffix[e + 1::], next, next + 1)


def url_param2value(param):
    """
    CONVERT URL QUERY PARAMETERS INTO DICT
    """
    def _decode(v):
        output = []
        i = 0
        while i < len(v):
            c = v[i]
            if c == "%":
                d = _url_to_str[v[i:i + 3]]
                output.append(d)
                i += 3
            else:
                output.append(c)
                i += 1

        output = ("".join(output))
        try:
            if not _Log:
                _late_import()
            return _json2value(output)
        except Exception:
            pass
        return output

    query = Data()
    for p in param.split("&"):
        if not p:
            continue
        if p.find("=") == -1:
            k = p
            v = True
        else:
            k, v = p.split("=")
            v = _decode(v)

        u = query.get(k)
        if u is None:
            query[k] = v
        elif isinstance(u, list):
            u += [v]
        else:
            query[k] = [u, v]

    return query




def value2url_param(value):
    """
    :param value:
    :return: ascii URL
    """
    if not _Log:
        _late_import()

    if value == None:
        _Log.error(u"Can not encode None into a URL")

    if isinstance(value, Mapping):
        value_ = wrap(value)
        output = "&".join([
            value2url_param(k) + "=" + (value2url_param(v) if isinstance(v, text_type) else value2url_param(_value2json(v)))
            for k, v in value_.leaves()
            ])
    elif isinstance(value, text_type):
        output = "".join(_str_to_url[c] for c in value)
    elif hasattr(value, "__iter__"):
        output = ",".join(value2url_param(v) for v in value)
    else:
        output = str(value)
    return output




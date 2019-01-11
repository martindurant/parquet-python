# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from mo_dots import Data, Null, coalesce, is_data, is_list, wrap
from mo_future import PY2, is_text, text_type, unichr, urlparse, is_binary
from mo_json import json2value, value2json
from mo_logs import Log


class URL(object):
    """
    JUST LIKE urllib.parse() [1], BUT CAN HANDLE JSON query PARAMETERS

    [1] https://docs.python.org/3/library/urllib.parse.html
    """

    def __init__(self, value, port=None, path=None, query=None, fragment=None):
        try:
            self.scheme = None
            self.host = None
            self.port = port
            self.path = path
            self.query = query
            self.fragment = fragment

            if value == None:
                return

            if value.startswith("file://") or value.startswith("//"):
                # urlparse DOES NOT WORK IN THESE CASES
                scheme, suffix = value.split("//", 2)
                self.scheme = scheme.rstrip(":")
                parse(self, suffix, 0, 1)
                self.query = wrap(url_param2value(self.query))
            else:
                output = urlparse(value)
                self.scheme = output.scheme
                self.port = coalesce(port, output.port)
                self.host = output.netloc.split(":")[0]
                self.path = coalesce(path, output.path)
                self.query = coalesce(query, wrap(url_param2value(output.query)))
                self.fragment = coalesce(fragment, output.fragment)
        except Exception as e:
            Log.error(u"problem parsing {{value}} to URL", value=value, cause=e)

    def __nonzero__(self):
        if self.scheme or self.host or self.port or self.path or self.query or self.fragment:
            return True
        return False

    def __bool__(self):
        if self.scheme or self.host or self.port or self.path or self.query or self.fragment:
            return True
        return False

    def __truediv__(self, other):
        if not is_text(other):
            Log.error(u"Expecting text path")
        output = self.__copy__()
        output.path = output.path.rstrip('/') + "/" + other.lstrip('/')
        return output

    def __unicode__(self):
        return self.__str__().decode('utf8')  # ASSUME chr<128 ARE VALID UNICODE

    def __copy__(self):
        output = URL(None)
        output.scheme = self.scheme
        output.host = self.host
        output.port = self.port
        output.path = self.path
        output.query = self.query
        output.fragment = self.fragment
        return output

    def __data__(self):
        return str(self)

    def __str__(self):
        url = ""
        if self.host:
            url = self.host
        if self.scheme:
            url = self.scheme + "://"+url
        if self.port:
            url = url + ":" + str(self.port)
        if self.path:
            if self.path[0] == text_type("/"):
                url += str(self.path)
            else:
                url += "/" + str(self.path)
        if self.query:
            url = url + "?" + value2url_param(self.query)
        if self.fragment:
            url = url + "#" + value2url_param(self.fragment)
        return url


def int2hex(value, size):
    return (("0" * size) + hex(value)[2:])[-size:]


def hex2chr(hex):
    return unichr(int(hex, 16))


if PY2:
    _map2url = {chr(i): chr(i) for i in range(32, 128)}
    for c in " {}<>;/?:@&=+$,":
        _map2url[c] = "%" + str(int2hex(ord(c), 2))
    for i in range(128, 256):
        _map2url[chr(i)] = "%" + str(int2hex(i, 2))
else:
    _map2url = {i: unichr(i) for i in range(32, 128)}
    for c in b" {}<>;/?:@&=+$,":
        _map2url[c] = "%" + int2hex(c, 2)
    for i in range(128, 256):
        _map2url[i] = "%" + str(int2hex(i, 2))


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
    if param == None:
        return Null
    if param == None:
        return Null

    def _decode(v):
        output = []
        i = 0
        while i < len(v):
            c = v[i]
            if c == "%":
                d = hex2chr(v[i + 1:i + 3])
                output.append(d)
                i += 3
            else:
                output.append(c)
                i += 1

        output = text_type("".join(output))
        try:
            return json2value(output)
        except Exception:
            pass
        return output

    query = Data()
    for p in param.split('&'):
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
        elif is_list(u):
            u += [v]
        else:
            query[k] = [u, v]

    return query


def value2url_param(value):
    """
    :param value:
    :return: ascii URL
    """
    if value == None:
        Log.error("Can not encode None into a URL")

    if is_data(value):
        value_ = wrap(value)
        output = "&".join([
            value2url_param(k) + "=" + (value2url_param(v) if is_text(v) else value2url_param(value2json(v)))
            for k, v in value_.leaves()
            ])
    elif is_text(value):
        output = "".join(_map2url[c] for c in value.encode('utf8'))
    elif is_binary(value):
        output = "".join(_map2url[c] for c in value)
    elif hasattr(value, "__iter__"):
        output = ",".join(value2url_param(v) for v in value)
    else:
        output = str(value)
    return output




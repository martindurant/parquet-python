from collections import Mapping
from urlparse import urlparse

from mo_dots import wrap, Data
from mo_json import value2json, json2value
from mo_logs import Log


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
                scheme, suffix = value.split("//", 2)
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
            Log.error("problem parsing {{value}} to URL", value=value, cause=e)

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
        url = b""
        if self.host:
            url = self.host
        if self.scheme:
            url = self.scheme + b"://"+url
        if self.port:
            url = url + b":" + str(self.port)
        if self.path:
            if self.path[0]=="/":
                url += str(self.path)
            else:
                url += b"/"+str(self.path)
        if self.query:
            url = url + b'?' + value2url_param(self.query)
        if self.fragment:
            url = url + b'#' + value2url_param(self.fragment)
        return url


def int2hex(value, size):
    return (("0" * size) + hex(value)[2:])[-size:]


_map2url = {chr(i): chr(i) for i in range(32, 128)}
for c in b" {}<>;/?:@&=+$,":
    _map2url[c] = b"%" + str(int2hex(ord(c), 2))
for i in range(128, 256):
    _map2url[chr(i)] = b"%" + str(int2hex(i, 2))


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
    if isinstance(param, text_type):
        param = param.encode("ascii")

    def _decode(v):
        output = []
        i = 0
        while i < len(v):
            c = v[i]
            if c == "%":
                d = (v[i + 1:i + 3]).decode("hex")
                output.append(d)
                i += 3
            else:
                output.append(c)
                i += 1

        output = (b"".join(output)).decode("latin1")
        try:
            return json2value(output)
        except Exception:
            pass
        return output

    query = Data()
    for p in param.split(b'&'):
        if not p:
            continue
        if p.find(b"=") == -1:
            k = p
            v = True
        else:
            k, v = p.split(b"=")
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
    if value == None:
        Log.error("Can not encode None into a URL")

    if isinstance(value, Mapping):
        value_ = wrap(value)
        output = b"&".join([
            value2url_param(k) + b"=" + (value2url_param(v) if isinstance(v, text_type) else value2url_param(value2json(v)))
            for k, v in value_.leaves()
            ])
    elif isinstance(value, text_type):
        output = b"".join(_map2url[c] for c in value.encode('utf8'))
    elif isinstance(value, str):
        output = b"".join(_map2url[c] for c in value)
    elif hasattr(value, "__iter__"):
        output = b",".join(value2url_param(v) for v in value)
    else:
        output = str(value)
    return output




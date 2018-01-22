# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

# MIMICS THE requests API (http://docs.python-requests.org/en/latest/)
# DEMANDS data IS A JSON-SERIALIZABLE STRUCTURE
# WITH ADDED default_headers THAT CAN BE SET USING mo_logs.settings
# EG
# {"debug.constants":{
#     "pyLibrary.env.http.default_headers":{"From":"klahnakoski@mozilla.com"}
# }}


from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from copy import copy
from mmap import mmap
from numbers import Number
from tempfile import TemporaryFile

from mo_future import text_type
from jx_python import jx
from mo_dots import Data, coalesce, wrap, set_default, unwrap
from mo_json import value2json
from mo_logs import Log
from mo_math import Math
from mo_threads import Lock
from mo_threads import Till
from pyLibrary import convert
from requests import sessions, Response

import mo_json
from mo_logs.exceptions import Except
from mo_times.durations import Duration
from pyLibrary.env.big_data import safe_size, ibytes2ilines, icompressed2ibytes

DEBUG = False
FILE_SIZE_LIMIT = 100 * 1024 * 1024
MIN_READ_SIZE = 8 * 1024
ZIP_REQUEST = False
default_headers = Data()  # TODO: MAKE THIS VARIABLE A SPECIAL TYPE OF EXPECTED MODULE PARAMETER SO IT COMPLAINS IF NOT SET
default_timeout = 600

_warning_sent = False


def request(method, url, zip=None, retry=None, **kwargs):
    """
    JUST LIKE requests.request() BUT WITH DEFAULT HEADERS AND FIXES
    DEMANDS data IS ONE OF:
    * A JSON-SERIALIZABLE STRUCTURE, OR
    * LIST OF JSON-SERIALIZABLE STRUCTURES, OR
    * None

    Parameters
     * zip - ZIP THE REQUEST BODY, IF BIG ENOUGH
     * json - JSON-SERIALIZABLE STRUCTURE
     * retry - {"times": x, "sleep": y} STRUCTURE

    THE BYTE_STRINGS (b"") ARE NECESSARY TO PREVENT httplib.py FROM **FREAKING OUT**
    IT APPEARS requests AND httplib.py SIMPLY CONCATENATE STRINGS BLINDLY, WHICH
    INCLUDES url AND headers
    """
    global _warning_sent
    if not default_headers and not _warning_sent:
        _warning_sent = True
        Log.warning(
            "The pyLibrary.env.http module was meant to add extra "
            "default headers to all requests, specifically the 'Referer' "
            "header with a URL to the project. Use the `pyLibrary.debug.constants.set()` "
            "function to set `pyLibrary.env.http.default_headers`"
        )

    if isinstance(url, list):
        # TRY MANY URLS
        failures = []
        for remaining, u in jx.countdown(url):
            try:
                response = request(method, u, zip=zip, retry=retry, **kwargs)
                if Math.round(response.status_code, decimal=-2) not in [400, 500]:
                    return response
                if not remaining:
                    return response
            except Exception as e:
                e = Except.wrap(e)
                failures.append(e)
        Log.error("Tried {{num}} urls", num=len(url), cause=failures)

    if b"session" in kwargs:
        session = kwargs[b"session"]
        del kwargs[b"session"]
    else:
        session = sessions.Session()
    session.headers.update(default_headers)

    if zip is None:
        zip = ZIP_REQUEST

    if isinstance(url, text_type):
        # httplib.py WILL **FREAK OUT** IF IT SEES ANY UNICODE
        url = url.encode("ascii")

    _to_ascii_dict(kwargs)
    timeout = kwargs[b'timeout'] = coalesce(kwargs.get(b'timeout'), default_timeout)

    if retry == None:
        retry = Data(times=1, sleep=0)
    elif isinstance(retry, Number):
        retry = Data(times=retry, sleep=1)
    else:
        retry = wrap(retry)
        if isinstance(retry.sleep, Duration):
            retry.sleep = retry.sleep.seconds
        set_default(retry, {"times": 1, "sleep": 0})

    if b'json' in kwargs:
        kwargs[b'data'] = value2json(kwargs[b'json']).encode("utf8")
        del kwargs[b'json']

    try:
        headers = kwargs[b"headers"] = unwrap(coalesce(kwargs.get(b'headers'), {}))
        set_default(headers, {b"Accept-Encoding": b"compress, gzip"})

        if zip and len(coalesce(kwargs.get(b"data"))) > 1000:
            compressed = convert.bytes2zip(kwargs[b"data"])
            headers[b'content-encoding'] = b'gzip'
            kwargs[b"data"] = compressed

            _to_ascii_dict(headers)
        else:
            _to_ascii_dict(headers)
    except Exception as e:
        Log.error("Request setup failure on {{url}}", url=url, cause=e)

    errors = []
    for r in range(retry.times):
        if r:
            Till(seconds=retry.sleep).wait()

        try:
            if DEBUG:
                Log.note("http {{method}} to {{url}}", method=method, url=url)
            return session.request(method=method, url=url, **kwargs)
        except Exception as e:
            errors.append(Except.wrap(e))

    if " Read timed out." in errors[0]:
        Log.error("Tried {{times}} times: Timeout failure (timeout was {{timeout}}", timeout=timeout, times=retry.times, cause=errors[0])
    else:
        Log.error("Tried {{times}} times: Request failure of {{url}}", url=url, times=retry.times, cause=errors[0])


def _to_ascii_dict(headers):
    if headers is None:
        return
    for k, v in copy(headers).items():
        if isinstance(k, text_type):
            del headers[k]
            if isinstance(v, text_type):
                headers[k.encode("ascii")] = v.encode("ascii")
            else:
                headers[k.encode("ascii")] = v
        elif isinstance(v, text_type):
            headers[k] = v.encode("ascii")


def get(url, **kwargs):
    kwargs.setdefault(b'allow_redirects', True)
    kwargs[b"stream"] = True
    return HttpResponse(request(b'get', url, **kwargs))


def get_json(url, **kwargs):
    """
    ASSUME RESPONSE IN IN JSON
    """
    response = get(url, **kwargs)
    c = response.all_content
    return mo_json.json2value(convert.utf82unicode(c))

def options(url, **kwargs):
    kwargs.setdefault(b'allow_redirects', True)
    kwargs[b"stream"] = True
    return HttpResponse(request(b'options', url, **kwargs))


def head(url, **kwargs):
    kwargs.setdefault(b'allow_redirects', False)
    kwargs[b"stream"] = True
    return HttpResponse(request(b'head', url, **kwargs))


def post(url, **kwargs):
    kwargs[b"stream"] = True
    return HttpResponse(request(b'post', url, **kwargs))


def delete(url, **kwargs):
    return HttpResponse(request(b'delete', url, **kwargs))


def post_json(url, **kwargs):
    """
    ASSUME RESPONSE IN IN JSON
    """
    if b"json" in kwargs:
        kwargs[b"data"] = convert.unicode2utf8(value2json(kwargs[b"json"]))
    elif b'data' in kwargs:
        kwargs[b"data"] = convert.unicode2utf8(value2json(kwargs[b"data"]))
    else:
        Log.error("Expecting `json` parameter")

    response = post(url, **kwargs)
    c = response.content
    try:
        details = mo_json.json2value(convert.utf82unicode(c))
    except Exception as e:
        Log.error("Unexpected return value {{content}}", content=c, cause=e)

    if response.status_code not in [200, 201]:
        Log.error("Bad response", cause=Except.wrap(details))

    return details


def put(url, **kwargs):
    return HttpResponse(request(b'put', url, **kwargs))


def patch(url, **kwargs):
    kwargs[b"stream"] = True
    return HttpResponse(request(b'patch', url, **kwargs))


def delete(url, **kwargs):
    kwargs[b"stream"] = True
    return HttpResponse(request(b'delete', url, **kwargs))


class HttpResponse(Response):
    def __new__(cls, resp):
        resp.__class__ = HttpResponse
        return resp

    def __init__(self, resp):
        pass
        self._cached_content = None

    @property
    def all_content(self):
        # response.content WILL LEAK MEMORY (?BECAUSE OF PYPY"S POOR HANDLING OF GENERATORS?)
        # THE TIGHT, SIMPLE, LOOP TO FILL blocks PREVENTS THAT LEAK
        if self._content is not False:
            self._cached_content = self._content
        elif self._cached_content is None:
            def read(size):
                if self.raw._fp.fp is not None:
                    return self.raw.read(amt=size, decode_content=True)
                else:
                    self.close()
                    return None

            self._cached_content = safe_size(Data(read=read))

        if hasattr(self._cached_content, "read"):
            self._cached_content.seek(0)

        return self._cached_content

    @property
    def all_lines(self):
        return self.get_all_lines()

    def get_all_lines(self, encoding="utf8", flexible=False):
        try:
            iterator = self.raw.stream(4096, decode_content=False)

            if self.headers.get('content-encoding') == 'gzip':
                return ibytes2ilines(icompressed2ibytes(iterator), encoding=encoding, flexible=flexible)
            elif self.headers.get('content-type') == 'application/zip':
                return ibytes2ilines(icompressed2ibytes(iterator), encoding=encoding, flexible=flexible)
            elif self.url.endswith(".gz"):
                return ibytes2ilines(icompressed2ibytes(iterator), encoding=encoding, flexible=flexible)
            else:
                return ibytes2ilines(iterator, encoding=encoding, flexible=flexible, closer=self.close)
        except Exception as e:
            Log.error("Can not read content", cause=e)


class Generator_usingStream(object):
    """
    A BYTE GENERATOR USING A STREAM, AND BUFFERING IT FOR RE-PLAY
    """

    def __init__(self, stream, length, _shared=None):
        """
        :param stream:  THE STREAM WE WILL GET THE BYTES FROM
        :param length:  THE MAX NUMBER OF BYTES WE ARE EXPECTING
        :param _shared: FOR INTERNAL USE TO SHARE THE BUFFER
        :return:
        """
        self.position = 0
        file_ = TemporaryFile()
        if not _shared:
            self.shared = Data(
                length=length,
                locker=Lock(),
                stream=stream,
                done_read=0,
                file=file_,
                buffer=mmap(file_.fileno(), length)
            )
        else:
            self.shared = _shared

        self.shared.ref_count += 1

    def __iter__(self):
        return Generator_usingStream(None, self.shared.length, self.shared)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def next(self):
        if self.position >= self.shared.length:
            raise StopIteration

        end = min(self.position + MIN_READ_SIZE, self.shared.length)
        s = self.shared
        with s.locker:
            while end > s.done_read:
                data = s.stream.read(MIN_READ_SIZE)
                s.buffer.write(data)
                s.done_read += MIN_READ_SIZE
                if s.done_read >= s.length:
                    s.done_read = s.length
                    s.stream.close()
        try:
            return s.buffer[self.position:end]
        finally:
            self.position = end

    def close(self):
        with self.shared.locker:
            if self.shared:
                s, self.shared = self.shared, None
                s.ref_count -= 1

                if s.ref_count==0:
                    try:
                        s.stream.close()
                    except Exception:
                        pass

                    try:
                        s.buffer.close()
                    except Exception:
                        pass

                    try:
                        s.file.close()
                    except Exception:
                        pass

    def __del__(self):
        self.close()

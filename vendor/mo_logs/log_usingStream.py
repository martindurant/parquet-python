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
import sys

from mo_future import PY3, allocate_lock
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import CR, expand_template


class StructuredLogger_usingStream(StructuredLogger):
    def __init__(self, stream):
        try:
            self.locker = allocate_lock()
            self.flush = stream.flush
            if stream in (sys.stdout, sys.stderr):
                if PY3:
                    stream = stream.buffer
            self.writer = _UTF8Encoder(stream).write
        except Exception as _:
            sys.stderr.write("can not handle")

    def write(self, template, params):
        value = expand_template(template, params)
        self.locker.acquire()
        try:
            self.writer(value + CR)
        finally:
            self.locker.release()

    def stop(self):
        self.flush()


class _UTF8Encoder(object):

    def __init__(self, stream):
        self.stream = stream

    def write(self, v):
        try:
            self.stream.write(v.encode('utf8'))
        except Exception:
            sys.stderr.write("can not handle")

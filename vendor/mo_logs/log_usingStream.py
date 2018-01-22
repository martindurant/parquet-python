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

import sys

from mo_future import PY3, allocate_lock
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import expand_template


class StructuredLogger_usingStream(StructuredLogger):
    def __init__(self, stream):
        self.locker = allocate_lock()
        try:
            if stream in (sys.stdout, sys.stderr):
                if PY3:
                    self.writer = stream.write
                else:
                    self.writer = _UTF8Encoder(stream).write
            elif hasattr(stream, 'encoding') and stream.encoding:
                self.writer = _UTF8Encoder(stream).write
            else:
                self.writer = stream.write
        except Exception as e:
            sys.stderr("can not handle")

    def write(self, template, params):
        value = expand_template(template, params)
        self.locker.acquire()
        try:
            self.writer(value + "\n")
        finally:
            self.locker.release()

    def stop(self):
        pass


class _UTF8Encoder(object):

    def __init__(self, stream):
        self.stream = stream

    def write(self, v):
        try:
            self.stream.write(v.encode('utf8'))
        except Exception as e:
            sys.stderr("can not handle")

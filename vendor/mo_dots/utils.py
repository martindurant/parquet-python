# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

import importlib
import sys

from mo_future import PY2, text_type

OBJ = text_type("_obj")
CLASS = text_type("__class__")

_Log = None

if PY2:
    STDOUT = sys.stdout
    STDERR = sys.stderr
else:
    STDOUT = sys.stdout.buffer
    STDERR = sys.stderr.buffer


def get_logger():
    global _Log
    if _Log:
        return _Log
    try:
        from mo_logs import Log as _Log
        return _Log
    except Exception as e:
        _Log = PoorLogger()
        _Log.warning("`pip install mo-logs` for better logging.", cause=e)
        return _Log



def get_module(name):
    try:
        return importlib.import_module(name)
    except Exception as e:
        get_logger().error("`pip install " + name.split(".")[0].replace("_", "-") + "` to enable this feature", cause=e)


class PoorLogger(object):
    @classmethod
    def note(cls, note, **kwargs):
        STDOUT.write(note.encode('utf8')+b"\n")

    @classmethod
    def warning(cls, note, **kwargs):
        STDOUT.write(b"WARNING: " + note.encode('utf8') + b"\n")

    @classmethod
    def error(cls, note, **kwargs):
        STDERR.write(note.encode('utf8'))
        if str("cause") in kwargs:
            raise kwargs[str("cause")]
        else:
            raise Exception(note)


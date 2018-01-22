# encoding: utf-8
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

import importlib
import sys


_Log = None


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
    def note(self, note, **kwargs):
        sys.stdout.write(note+"\n")

    def warning(self, note, **kwargs):
        sys.stdout.write("WARNING: "+note+"\n")

    def error(self, note, **kwargs):
        sys.stderr.write(note)
        if "cause" in kwargs:
            raise kwargs["cause"]
        else:
            raise Exception(note)


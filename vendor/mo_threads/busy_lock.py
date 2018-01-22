
# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
# THIS THREADING MODULE IS PERMEATED BY THE please_stop SIGNAL.
# THIS SIGNAL IS IMPORTANT FOR PROPER SIGNALLING WHICH ALLOWS
# FOR FAST AND PREDICTABLE SHUTDOWN AND CLEANUP OF THREADS

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from time import sleep

from mo_future import get_ident as _get_ident

_get = dict.get
_set = dict.setdefault


class BusyLock(object):
    """
    ONLY USE IF HOLDING THE LOCK FOR A SHORT TIME
    """

    def __init__(self):
        self.lock = {}

    def __enter__(self):
        id = _get_ident()
        lock = self.lock

        while True:
            v = _get(lock, 0)
            if not v:
                _set(lock, 0, id)
            elif v == id:
                break
            else:
                sleep(0.000001)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lock[0]

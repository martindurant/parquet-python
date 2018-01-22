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

from mo_future import allocate_lock as _allocate_lock
from mo_threads.signal import Signal

_Log = None
_Except = None
_Thread = None
_extract_stack = None

DEBUG = False
DEBUG_SIGNAL = False


def _late_import():
    global _Log
    global _Except
    global _Thread
    global _extract_stack

    if _Thread:
        return

    from mo_logs.exceptions import Except as _Except
    from mo_logs.exceptions import extract_stack as _extract_stack
    from mo_threads.threads import Thread as _Thread
    from mo_logs import Log as _Log

    _ = _Log
    _ = _Except
    _ = _Thread
    _ = _extract_stack


class Lock(object):
    """
    A NON-RE-ENTRANT LOCK WITH wait() AND
    """
    __slots__ = ["name", "lock", "waiting"]

    def __init__(self, name=""):
        if DEBUG and not _Log:
            _late_import()
        self.name = name
        self.lock = _allocate_lock()
        self.waiting = None

    def __enter__(self):
        # with mo_times.timer.Timer("get lock"):
        self.lock.acquire()
        return self

    def __exit__(self, a, b, c):
        if self.waiting:
            if DEBUG:
                _Log.note("signaling {{num}} waiters", num=len(self.waiting))
            waiter = self.waiting.pop()
            waiter.go()
        self.lock.release()

    def wait(self, till=None):
        """
        THE ASSUMPTION IS wait() WILL ALWAYS RETURN WITH THE LOCK ACQUIRED
        :param till: WHEN TO GIVE UP WAITING FOR ANOTHER THREAD TO SIGNAL
        :return: True IF SIGNALED TO GO, False IF TIMEOUT HAPPENED
        """
        waiter = Signal()
        if self.waiting:
            if DEBUG:
                _Log.note("waiting with {{num}} others on {{name|quote}}", num=len(self.waiting), name=self.name)
            self.waiting.insert(0, waiter)
        else:
            if DEBUG:
                _Log.note("waiting by self on {{name|quote}}", name=self.name)
            self.waiting = [waiter]

        try:
            self.lock.release()
            if DEBUG:
                _Log.note("out of lock {{name|quote}}", name=self.name)
            (waiter | till).wait()
            if DEBUG:
                _Log.note("done minimum wait (for signal {{till|quote}})", till=till.name if till else "", name=self.name)
        except Exception as e:
            if not _Log:
                _late_import()
            _Log.warning("problem", cause=e)
        finally:
            self.lock.acquire()
            if DEBUG:
                _Log.note("re-acquired lock {{name|quote}}", name=self.name)

        try:
            self.waiting.remove(waiter)
            if DEBUG:
                _Log.note("removed own signal from {{name|quote}}", name=self.name)
        except Exception:
            pass

        return bool(waiter)

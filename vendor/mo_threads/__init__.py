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

from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
from mo_threads.lock import Lock
from mo_threads.multiprocess import Process
from mo_threads.queues import Queue, ThreadedQueue
from mo_threads.signal import Signal
from mo_threads.threads import MAIN_THREAD, MainThread, THREAD_STOP, THREAD_TIMEOUT, Thread, stop_main_thread
from mo_threads.till import Till

MAIN_THREAD.timers = Thread.run("timers daemon", till.daemon)
MAIN_THREAD.children.remove(threads.MAIN_THREAD.timers)





# from threading import Thread as _threading_Thread
# _temp = _threading_Thread.setDaemon
#
# fixes = []
# # WE NOW ADD A FIX FOR EACH KNOWN BAD ACTOR
# try:
#     from paramiko import Transport
#
#     def fix(self):
#         if isinstance(self, Transport):
#             self.stop = self.close   # WE KNOW Transport DOES NOT HAVE A stop() METHOD, SO ADDING SHOULD BE FINE
#             parent = Thread.current()
#             parent.add_child(self)
#             return True
#
#     fixes.append(fix)
# except Exception:
#     pass
#
#
# _known_daemons = [
#     ('thread_handling', 17),  # fabric/thread_handling.py
#     ('pydevd_comm.py', 285),  # plugins/python/helpers/pydev/_pydevd_bundle/pydevd_comm.py",
# ]
#
#
# # WE WRAP THE setDaemon METHOD TO APPLY THE FIX WHEN CALLED
# def _setDaemon(self, daemonic):
#     for fix in fixes:
#         if fix(self):
#             break
#     else:
#         from mo_logs import Log
#         from mo_logs.exceptions import extract_stack
#         from mo_files import File
#
#         get_function_name(self.__target)
#
#         stack = extract_stack(1)[0]
#         uid = (File(stack['file']).name, stack['line'])
#         if uid in _known_daemons:
#             pass
#         else:
#             _known_daemons.append(uid)
#             Log.warning("daemons in threading.Thread do not shutdown clean.  {{type}} not handled.", type=repr(self))
#
#     _temp(self, daemonic)
#
#
# _threading_Thread.setDaemon = _setDaemon
#
#


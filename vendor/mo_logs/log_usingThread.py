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

from mo_logs.exceptions import suppress_exception, Except
from mo_logs import Log
from mo_logs.log_usingNothing import StructuredLogger
from mo_threads import Thread, Queue, Till, THREAD_STOP


class StructuredLogger_usingThread(StructuredLogger):

    def __init__(self, logger):
        if not isinstance(logger, StructuredLogger):
            Log.error("Expecting a StructuredLogger")

        self.queue = Queue("Queue for " + self.__class__.__name__, max=10000, silent=True, allow_add_after_close=True)
        self.logger = logger

        def worker(logger, please_stop):
            try:
                while not please_stop:
                    Till(seconds=1).wait()
                    logs = self.queue.pop_all()
                    for log in logs:
                        if log is THREAD_STOP:
                            please_stop.go()
                        else:
                            logger.write(**log)
            finally:
                logger.stop()

        self.thread = Thread("Thread for " + self.__class__.__name__, worker, logger)
        self.thread.parent.remove_child(self.thread)  # LOGGING WILL BE RESPONSIBLE FOR THREAD stop()
        self.thread.start()

    def write(self, template, params):
        try:
            self.queue.add({"template": template, "params": params})
            return self
        except Exception as e:
            e = Except.wrap(e)
            raise e  # OH NO!

    def stop(self):
        with suppress_exception:
            self.queue.add(THREAD_STOP)  # BE PATIENT, LET REST OF MESSAGE BE SENT
            self.thread.join()
            self.logger.stop()

        with suppress_exception:
            self.queue.close()


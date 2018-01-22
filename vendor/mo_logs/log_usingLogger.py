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

import logging

from mo_logs import Log
from mo_logs.exceptions import suppress_exception
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.log_usingThreadedStream import StructuredLogger_usingThreadedStream, time_delta_pusher
from mo_dots import unwrap


_THREAD_STOP = None
_Queue = None
_Thread = None


def _late_import():
    global _THREAD_STOP
    global _Queue
    global _Thread

    from mo_threads import THREAD_STOP as _THREAD_STOP
    from mo_threads import Queue as _Queue
    from mo_threads import Thread as _Thread

    _ = _THREAD_STOP
    _ = _Queue
    _ = _Thread


# WRAP PYTHON CLASSIC logger OBJECTS
class StructuredLogger_usingLogger(StructuredLogger):
    def __init__(self, settings):
        if not _Thread:
            _late_import()

        self.logger = logging.Logger("unique name", level=logging.INFO)
        self.logger.addHandler(make_log_from_settings(settings))

        # TURNS OUT LOGGERS ARE REALLY SLOW TOO
        self.queue = _Queue("queue for classic logger", max=10000, silent=True)
        self.thread = _Thread(
            "pushing to classic logger",
            time_delta_pusher,
            appender=self.logger.info,
            queue=self.queue,
            interval=0.3
        )
        self.thread.parent.remove_child(self.thread)  # LOGGING WILL BE RESPONSIBLE FOR THREAD stop()
        self.thread.start()

    def write(self, template, params):
        # http://docs.python.org/2/library/logging.html# logging.LogRecord
        self.queue.add({"template": template, "params": params})

    def stop(self):
        with suppress_exception:
            self.queue.add(_THREAD_STOP)  # BE PATIENT, LET REST OF MESSAGE BE SENT
            self.thread.join()

        with suppress_exception:
            self.queue.close()


def make_log_from_settings(settings):
    assert settings["class"]

    # IMPORT MODULE FOR HANDLER
    path = settings["class"].split(".")
    class_name = path[-1]
    path = ".".join(path[:-1])
    constructor = None
    try:
        temp = __import__(path, globals(), locals(), [class_name], -1)
        constructor = object.__getattribute__(temp, class_name)
    except Exception as e:
        if settings.stream and not constructor:
            # PROVIDE A DEFAULT STREAM HANLDER
            constructor = StructuredLogger_usingThreadedStream
        else:
            Log.error("Can not find class {{class}}",  {"class": path}, cause=e)

    # IF WE NEED A FILE, MAKE SURE DIRECTORY EXISTS
    if settings.filename:
        from mo_files import File

        f = File(settings.filename)
        if not f.parent.exists:
            f.parent.create()

    settings['class'] = None
    params = unwrap(settings)
    log_instance = constructor(**params)
    return log_instance


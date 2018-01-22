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

from datetime import timedelta
from time import time

from mo_dots import coalesce, wrap
from mo_logs import Log

from mo_times.durations import Duration


class Timer(object):
    """
    USAGE:
    with Timer("doing hard time"):
        something_that_takes_long()
    OUTPUT:
        doing hard time took 45.468 sec

    param - USED WHEN LOGGING
    debug - SET TO False TO DISABLE THIS TIMER
    """

    def __init__(self, description, param=None, debug=True, silent=False):
        self.template = description
        self.param = wrap(coalesce(param, {}))
        self.debug = debug
        self.silent = silent
        self.start = 0
        self.end = 0
        self.interval = None

    def __enter__(self):
        if self.debug:
            if not self.silent:
                Log.note("Timer start: " + self.template, stack_depth=1, **self.param)
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time()
        self.interval = self.end - self.start

        if self.debug:
            param = wrap(self.param)
            param.duration = timedelta(seconds=self.interval)
            if not self.silent:
                Log.note("Timer end  : " + self.template + " (took {{duration}})", self.param, stack_depth=1)

    @property
    def duration(self):
        if not self.end:
            return Duration(time() - self.start)

        return Duration(self.interval)

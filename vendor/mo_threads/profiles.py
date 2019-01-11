# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
import cProfile
from datetime import datetime
import pstats

from mo_future import iteritems
from mo_logs import Log

FILENAME = "profile.tab"

cprofiler_stats = None  # ACCUMULATION OF STATS FROM ALL THREADS


class CProfiler(object):
    """
    cProfiler CONTEXT MANAGER WRAPPER
    """

    __slots__ = ["cprofiler"]

    def __init__(self):
        self.cprofiler = None

    def __enter__(self):
        if cprofiler_stats is not None:
            Log.note("starting cprofile")
            self.cprofiler = cProfile.Profile()
            self.cprofiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cprofiler is not None:
            self.cprofiler.disable()
            cprofiler_stats.add(pstats.Stats(self.cprofiler))
            del self.cprofiler
            Log.note("done cprofile")

    def enable(self):
        if self.cprofiler is not None:
            return self.cprofiler.enable()

    def disable(self):
        if self.cprofiler is not None:
            return self.cprofiler.disable()


def enable_profilers(filename):
    global FILENAME
    global cprofiler_stats

    if cprofiler_stats is not None:
        return
    if filename:
        FILENAME = filename

    from mo_threads.threads import ALL_LOCK, ALL, Thread
    from mo_threads.queues import Queue
    cprofiler_stats = Queue("cprofiler stats")

    current_thread = Thread.current()
    with ALL_LOCK:
        threads = list(ALL.values())
    for t in threads:
        t.cprofiler = CProfiler()
        if t is current_thread:
            Log.note("starting cprofile for thread {{name}}", name=t.name)
            t.cprofiler.__enter__()
        else:
            Log.note("cprofiler not started for thread {{name}} (already running)", name=t.name)


def write_profiles(main_thread_profile):
    if cprofiler_stats is None:
        return

    from pyLibrary import convert
    from mo_files import File

    cprofiler_stats.add(pstats.Stats(main_thread_profile.cprofiler))
    stats = cprofiler_stats.pop_all()

    Log.note("aggregating {{num}} profile stats", num=len(stats))
    acc = stats[0]
    for s in stats[1:]:
        acc.add(s)

    stats = [
        {
            "num_calls": d[1],
            "self_time": d[2],
            "total_time": d[3],
            "self_time_per_call": d[2] / d[1],
            "total_time_per_call": d[3] / d[1],
            "file": (f[0] if f[0] != "~" else "").replace("\\", "/"),
            "line": f[1],
            "method": f[2].lstrip("<").rstrip(">")
        }
        for f, d, in iteritems(acc.stats)
    ]
    stats_file = File(FILENAME, suffix=convert.datetime2string(datetime.now(), "_%Y%m%d_%H%M%S"))
    stats_file.write(convert.list2tab(stats))
    Log.note("profile written to {{filename}}", filename=stats_file.abspath)

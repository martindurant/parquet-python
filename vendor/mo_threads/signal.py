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
import random
from weakref import ref

from mo_future import allocate_lock as _allocate_lock, text_type
from mo_logs import Log

DEBUG = False
DEBUG_SIGNAL = False
SEED = random.Random()



class Signal(object):
    """
    SINGLE-USE THREAD SAFE SIGNAL

    go() - ACTIVATE SIGNAL (DOES NOTHING IF SIGNAL IS ALREADY ACTIVATED)
    wait() - PUT THREAD IN WAIT STATE UNTIL SIGNAL IS ACTIVATED
    on_go() - METHOD FOR OTHER THREAD TO RUN WHEN ACTIVATING SIGNAL
    """

    __slots__ = ["_name", "lock", "_go", "job_queue", "waiting_threads", "__weakref__"]

    def __init__(self, name=None):
        (DEBUG and name) and Log.note("New signal {{name|quote}}", name=name)
        self._name = name
        self.lock = _allocate_lock()
        self._go = False
        self.job_queue = None
        self.waiting_threads = None

    def __str__(self):
        return str(self._go)

    def __bool__(self):
        return self._go

    def __nonzero__(self):
        return self._go

    def wait(self):
        """
        PUT THREAD IN WAIT STATE UNTIL SIGNAL IS ACTIVATED
        """
        if self._go:
            return True

        with self.lock:
            if self._go:
                return True
            stopper = _allocate_lock()
            stopper.acquire()
            if not self.waiting_threads:
                self.waiting_threads = [stopper]
            else:
                self.waiting_threads.append(stopper)

        DEBUG and self._name and Log.note("wait for go {{name|quote}}", name=self.name)
        stopper.acquire()
        DEBUG and self._name and Log.note("GOing! {{name|quote}}", name=self.name)
        return True

    def go(self):
        """
        ACTIVATE SIGNAL (DOES NOTHING IF SIGNAL IS ALREADY ACTIVATED)
        """
        DEBUG and self._name and Log.note("GO! {{name|quote}}", name=self.name)

        if self._go:
            return

        with self.lock:
            if self._go:
                return
            self._go = True

        DEBUG and self._name and Log.note("internal GO! {{name|quote}}", name=self.name)
        jobs, self.job_queue = self.job_queue, None
        threads, self.waiting_threads = self.waiting_threads, None

        if threads:
            DEBUG and self._name and Log.note("Release {{num}} threads", num=len(threads))
            for t in threads:
                t.release()

        if jobs:
            for j in jobs:
                try:
                    j()
                except Exception as e:
                    Log.warning("Trigger on Signal.go() failed!", cause=e)

    def on_go(self, target):
        """
        RUN target WHEN SIGNALED
        """
        if not target:
            Log.error("expecting target")

        with self.lock:
            if not self._go:
                DEBUG and self._name and Log.note("Adding target to signal {{name|quote}}", name=self.name)

                if not self.job_queue:
                    self.job_queue = [target]
                else:
                    self.job_queue.append(target)
                return

        (DEBUG_SIGNAL) and Log.note("Signal {{name|quote}} already triggered, running job immediately", name=self.name)
        target()

    def remove_go(self, target):
        """
        FOR SAVING MEMORY
        """
        with self.lock:
            if not self._go:
                try:
                    self.job_queue.remove(target)
                except ValueError:
                    pass

    @property
    def name(self):
        if not self._name:
            return "anonymous signal"
        else:
            return self._name

    def __str__(self):
        return self.name.decode(text_type)

    def __repr__(self):
        return text_type(repr(self._go))

    def __or__(self, other):
        if other == None:
            return self
        if not isinstance(other, Signal):
            Log.error("Expecting OR with other signal")

        output = Signal(self.name + " | " + other.name)
        OrSignal(output, (self, other))
        return output

    def __ror__(self, other):
        return self.__or__(other)

    def __and__(self, other):
        if other == None:
            return self
        if not isinstance(other, Signal):
            Log.error("Expecting OR with other signal")

        if DEBUG and self._name:
            output = Signal(self.name + " and " + other.name)
        else:
            output = Signal(self.name + " and " + other.name)

        gen = AndSignals(output, 2)
        self.on_go(gen.done)
        other.on_go(gen.done)
        return output


class AndSignals(object):
    __slots__ = ["signal", "remaining", "locker"]

    def __init__(self, signal, count):
        """
        CALL signal.go() WHEN done() IS CALLED count TIMES
        :param signal:
        :param count:
        :return:
        """
        self.signal = signal
        self.locker = _allocate_lock()
        self.remaining = count

    def done(self):
        with self.locker:
            self.remaining -= 1
            remaining = self.remaining
        if not remaining:
            self.signal.go()


class OrSignal(object):
    """
    A SELF-REFERENTIAL CLUSTER OF SIGNALING METHODS TO IMPLEMENT __or__()
    MANAGE SELF-REMOVAL UPON NOT NEEDING THE signal OBJECT ANY LONGER
    """
    __slots__ = ["signal", "dependencies"]

    def __init__(self, signal, dependencies):
        self.dependencies = dependencies
        self.signal = ref(signal, self.cleanup)
        for d in dependencies:
            d.on_go(self)
        signal.on_go(self.cleanup)

    def cleanup(self, r=None):
        for d in self.dependencies:
            d.remove_go(self)

    def __call__(self, *args, **kwargs):
        s = self.signal()
        if s is not None:
            s.go()


DONE = Signal()
DONE.go()

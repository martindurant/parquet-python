
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
from collections import deque
from datetime import datetime
from time import time
import types

from mo_dots import Null, coalesce
from mo_future import long
from mo_logs import Except, Log
from mo_threads.lock import Lock
from mo_threads.signal import Signal
from mo_threads.threads import THREAD_STOP, THREAD_TIMEOUT, Thread
from mo_threads.till import Till

DEBUG = False

# MAX_DATETIME = datetime(2286, 11, 20, 17, 46, 39)
DEFAULT_WAIT_TIME = 10 * 60  # SECONDS

datetime.strptime('2012-01-01', '%Y-%m-%d')  # http://bugs.python.org/issue7980


class Queue(object):
    """
     SIMPLE MESSAGE QUEUE, multiprocessing.Queue REQUIRES SERIALIZATION, WHICH
     IS DIFFICULT TO USE JUST BETWEEN THREADS (SERIALIZATION REQUIRED)
    """

    def __init__(self, name, max=None, silent=False, unique=False, allow_add_after_close=False):
        """
        max - LIMIT THE NUMBER IN THE QUEUE, IF TOO MANY add() AND extend() WILL BLOCK
        silent - COMPLAIN IF THE READERS ARE TOO SLOW
        unique - SET True IF YOU WANT ONLY ONE INSTANCE IN THE QUEUE AT A TIME
        """
        self.name = name
        self.max = coalesce(max, 2 ** 10)
        self.silent = silent
        self.allow_add_after_close=allow_add_after_close
        self.unique = unique
        self.closed = Signal("stop adding signal for " + name)  # INDICATE THE PRODUCER IS DONE GENERATING ITEMS TO QUEUE
        self.lock = Lock("lock for queue " + name)
        self.queue = deque()
        self.next_warning = time()  # FOR DEBUGGING

    def __iter__(self):
        try:
            while True:
                value = self.pop()
                if value is THREAD_STOP:
                    break
                if value is not None:
                    yield value
        except Exception as e:
            Log.warning("Tell me about what happened here", e)

    def add(self, value, timeout=None, force=False):
        """
        :param value:  ADDED THE THE QUEUE
        :param timeout:  HOW LONG TO WAIT FOR QUEUE TO NOT BE FULL
        :param force:  ADD TO QUEUE, EVEN IF FULL (USE ONLY WHEN CONSUMER IS RETURNING WORK TO THE QUEUE)
        :return: self
        """
        with self.lock:
            if value is THREAD_STOP:
                # INSIDE THE lock SO THAT EXITING WILL RELEASE wait()
                self.queue.append(value)
                self.closed.go()
                return

            if not force:
                self._wait_for_queue_space(timeout=timeout)
            if self.closed and not self.allow_add_after_close:
                Log.error("Do not add to closed queue")
            else:
                if self.unique:
                    if value not in self.queue:
                        self.queue.append(value)
                else:
                    self.queue.append(value)
        return self

    def push(self, value):
        """
        SNEAK value TO FRONT OF THE QUEUE
        """
        if self.closed and not self.allow_add_after_close:
            Log.error("Do not push to closed queue")

        with self.lock:
            self._wait_for_queue_space()
            if not self.closed:
                self.queue.appendleft(value)
        return self

    def pop_message(self, till=None):
        """
        RETURN TUPLE (message, payload) CALLER IS RESPONSIBLE FOR CALLING message.delete() WHEN DONE
        DUMMY IMPLEMENTATION FOR DEBUGGING
        """

        if till is not None and not isinstance(till, Signal):
            Log.error("Expecting a signal")
        return Null, self.pop(till=till)

    def extend(self, values):
        if self.closed and not self.allow_add_after_close:
            Log.error("Do not push to closed queue")

        with self.lock:
            # ONCE THE queue IS BELOW LIMIT, ALLOW ADDING MORE
            self._wait_for_queue_space()
            if not self.closed:
                if self.unique:
                    for v in values:
                        if v is THREAD_STOP:
                            self.closed.go()
                            continue
                        if v not in self.queue:
                            self.queue.append(v)
                else:
                    for v in values:
                        if v is THREAD_STOP:
                            self.closed.go()
                            continue
                        self.queue.append(v)
        return self

    def _wait_for_queue_space(self, timeout=DEFAULT_WAIT_TIME):
        """
        EXPECT THE self.lock TO BE HAD, WAITS FOR self.queue TO HAVE A LITTLE SPACE
        """
        wait_time = 5

        (DEBUG and len(self.queue) > 1 * 1000 * 1000) and Log.warning("Queue {{name}} has over a million items")

        now = time()
        if timeout != None:
            time_to_stop_waiting = now + timeout
        else:
            time_to_stop_waiting = now + DEFAULT_WAIT_TIME

        if self.next_warning < now:
            self.next_warning = now + wait_time

        while not self.closed and len(self.queue) >= self.max:
            if now > time_to_stop_waiting:
                Log.error(THREAD_TIMEOUT)

            if self.silent:
                self.lock.wait(Till(till=time_to_stop_waiting))
            else:
                self.lock.wait(Till(seconds=wait_time))
                if len(self.queue) >= self.max:
                    now = time()
                    if self.next_warning < now:
                        self.next_warning = now + wait_time
                        Log.alert(
                            "Queue by name of {{name|quote}} is full with ({{num}} items), thread(s) have been waiting {{wait_time}} sec",
                            name=self.name,
                            num=len(self.queue),
                            wait_time=wait_time
                        )

    def __len__(self):
        with self.lock:
            return len(self.queue)

    def __nonzero__(self):
        with self.lock:
            return any(r != THREAD_STOP for r in self.queue)

    def pop(self, till=None):
        """
        WAIT FOR NEXT ITEM ON THE QUEUE
        RETURN THREAD_STOP IF QUEUE IS CLOSED
        RETURN None IF till IS REACHED AND QUEUE IS STILL EMPTY

        :param till:  A `Signal` to stop waiting and return None
        :return:  A value, or a THREAD_STOP or None
        """
        if till is not None and not isinstance(till, Signal):
            Log.error("expecting a signal")

        with self.lock:
            while True:
                if self.queue:
                    value = self.queue.popleft()
                    return value
                if self.closed:
                    break
                if not self.lock.wait(till=self.closed | till):
                    if self.closed:
                        break
                    return None
        (DEBUG or not self.silent) and Log.note(self.name + " queue closed")
        return THREAD_STOP

    def pop_all(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        with self.lock:
            output = list(self.queue)
            self.queue.clear()

        return output

    def pop_one(self):
        """
        NON-BLOCKING POP IN QUEUE, IF ANY
        """
        with self.lock:
            if self.closed:
                return [THREAD_STOP]
            elif not self.queue:
                return None
            else:
                v =self.queue.pop()
                if v is THREAD_STOP:  # SENDING A STOP INTO THE QUEUE IS ALSO AN OPTION
                    self.closed.go()
                return v

    def close(self):
        self.closed.go()

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PriorityQueue(Queue):
    """
        ADDS ITEMS TO THEIR PRIORITY AND POP'S THE HIGHEST PRIORITY VALUE (UNLESS REQUESTED OTHERWISE)
    """
    def __init__(self, name, numpriorities, max=None, silent=False, unique=False, allow_add_after_close=False):
        Queue.__init__(self, name=name, max=max, silent=silent, unique=False, allow_add_after_close=False)

        self.numpriorities = numpriorities
        self.queue = [
            Queue(name=name, max=max, silent=silent, unique=False, allow_add_after_close=False)
            for _ in range(numpriorities)
        ]

    def __iter__(self):
        try:
            while True:
                value = self.pop(self.closed)
                if value is THREAD_STOP:
                    break
                if value is not None:
                    yield value
        except Exception as e:
            Log.warning("Tell me about what happened here", e)

        if not self.silent:
            Log.note("queue iterator is done")

    def add(self, value, timeout=None, priority=0):
        with self.lock:
            if value is THREAD_STOP:
                # INSIDE THE lock SO THAT EXITING WILL RELEASE wait()
                self.queue[priority].queue.append(value)
                self.closed.go()
                return

            self.queue[priority]._wait_for_queue_space(timeout=timeout)
            if self.closed and not self.queue[priority].allow_add_after_close:
                Log.error("Do not add to closed queue")
            else:
                if self.unique:
                    if value not in self.queue[priority].queue:
                        self.queue[priority].queue.append(value)
                else:
                    self.queue[priority].queue.append(value)
        return self

    def push(self, value, priority=0):
        """
        SNEAK value TO FRONT OF THE QUEUE
        """
        if self.closed and not self.queue[priority].allow_add_after_close:
            Log.error("Do not push to closed queue")

        with self.lock:
            self.queue[priority]._wait_for_queue_space()
            if not self.closed:
                self.queue[priority].queue.appendleft(value)
        return self

    def __len__(self):
        with self.lock:
            return sum([len(q.queue) for q in self.queue])

    def __nonzero__(self):
        with self.lock:
            return any(any(r != THREAD_STOP for r in q.queue) for q in self.queue)

    def highest_entry(self):
        for count, q in enumerate(self.queue):
            if len(q) > 0:
                return count
        return None

    def pop(self, till=None, priority=None):
        """
        WAIT FOR NEXT ITEM ON THE QUEUE
        RETURN THREAD_STOP IF QUEUE IS CLOSED
        RETURN None IF till IS REACHED AND QUEUE IS STILL EMPTY

        :param till:  A `Signal` to stop waiting and return None
        :return:  A value, or a THREAD_STOP or None
        """
        if till is not None and not isinstance(till, Signal):
            Log.error("expecting a signal")

        with self.lock:
            while True:
                if not priority:
                    priority = self.highest_entry()
                if priority:
                    value = self.queue[priority].queue.popleft()
                    return value
                if self.closed:
                    break
                if not self.lock.wait(till=till | self.closed):
                    if self.closed:
                        break
                    return None
        (DEBUG or not self.silent) and Log.note(self.name + " queue stopped")
        return THREAD_STOP

    def pop_all(self, priority=None):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        output = []
        with self.lock:
            if not priority:
                priority = self.highest_entry()
            if priority:
                output = list(self.queue[priority].queue)
                self.queue[priority].queue.clear()
        return output

    def pop_all_queues(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        output = []
        with self.lock:
            for q in self.queue:
                output.extend(list(q.queue))
                q.queue.clear()

        return output

    def pop_one(self, priority=None):
        """
        NON-BLOCKING POP IN QUEUE, IF ANY
        """
        with self.lock:
            if not priority:
                priority = self.highest_entry()
            if self.closed:
                return [THREAD_STOP]
            elif not self.queue:
                return None
            else:
                v =self.pop(priority=priority)
                if v is THREAD_STOP:  # SENDING A STOP INTO THE QUEUE IS ALSO AN OPTION
                    self.closed.go()
                return v



class ThreadedQueue(Queue):
    """
    DISPATCH TO ANOTHER (SLOWER) queue IN BATCHES OF GIVEN size
    TODO: Check that this queue is not dropping items at shutdown
    """

    def __init__(
        self,
        name,
        queue,  # THE SLOWER QUEUE
        batch_size=None,  # THE MAX SIZE OF BATCHES SENT TO THE SLOW QUEUE
        max_size=None,  # SET THE MAXIMUM SIZE OF THE QUEUE, WRITERS WILL BLOCK IF QUEUE IS OVER THIS LIMIT
        period=None,  # MAX TIME (IN SECONDS) BETWEEN FLUSHES TO SLOWER QUEUE
        silent=False,  # WRITES WILL COMPLAIN IF THEY ARE WAITING TOO LONG
        error_target=None  # CALL THIS WITH ERROR **AND THE LIST OF OBJECTS ATTEMPTED**
                           # BE CAREFUL!  THE THREAD MAKING THE CALL WILL NOT BE YOUR OWN!
                           # DEFAULT BEHAVIOUR: THIS WILL KEEP RETRYING WITH WARNINGS
    ):
        if period !=None and not isinstance(period, (int, float, long)):
            Log.error("Expecting a float for the period")

        batch_size = coalesce(batch_size, int(max_size / 2) if max_size else None, 900)
        max_size = coalesce(max_size, batch_size * 2)  # REASONABLE DEFAULT
        period = coalesce(period, 1)  # SECONDS

        Queue.__init__(self, name=name, max=max_size, silent=silent)

        def worker_bee(please_stop):
            please_stop.on_go(lambda: self.add(THREAD_STOP))

            _buffer = []
            _post_push_functions = []
            now = time()
            next_push = Till(till=now + period)  # THE TIME WE SHOULD DO A PUSH
            last_push = now - period

            def push_to_queue():
                queue.extend(_buffer)
                del _buffer[:]
                for ppf in _post_push_functions:
                    ppf()
                del _post_push_functions[:]

            while not please_stop:
                try:
                    if not _buffer:
                        item = self.pop()
                        now = time()
                        if now > last_push + period:
                            # Log.note("delay next push")
                            next_push = Till(till=now + period)
                    else:
                        item = self.pop(till=next_push)
                        now = time()

                    if item is THREAD_STOP:
                        push_to_queue()
                        please_stop.go()
                        break
                    elif isinstance(item, types.FunctionType):
                        _post_push_functions.append(item)
                    elif item is not None:
                        _buffer.append(item)

                except Exception as e:
                    e = Except.wrap(e)
                    if error_target:
                        try:
                            error_target(e, _buffer)
                        except Exception as f:
                            Log.warning(
                                "`error_target` should not throw, just deal",
                                name=name,
                                cause=f
                            )
                    else:
                        Log.warning(
                            "Unexpected problem",
                            name=name,
                            cause=e
                        )

                try:
                    if len(_buffer) >= batch_size or next_push:
                        if _buffer:
                            push_to_queue()
                            last_push = now = time()
                        next_push = Till(till=now + period)

                except Exception as e:
                    e = Except.wrap(e)
                    if error_target:
                        try:
                            error_target(e, _buffer)
                        except Exception as f:
                            Log.warning(
                                "`error_target` should not throw, just deal",
                                name=name,
                                cause=f
                            )
                    else:
                        Log.warning(
                            "Problem with {{name}} pushing {{num}} items to data sink",
                            name=name,
                            num=len(_buffer),
                            cause=e
                        )

            if _buffer:
                # ONE LAST PUSH, DO NOT HAVE TIME TO DEAL WITH ERRORS
                push_to_queue()

        self.thread = Thread.run("threaded queue for " + name, worker_bee) # parent_thread=self)

    def add(self, value, timeout=None):
        with self.lock:
            self._wait_for_queue_space(timeout=timeout)
            if not self.closed:
                self.queue.append(value)
        return self

    def extend(self, values):
        with self.lock:
            # ONCE THE queue IS BELOW LIMIT, ALLOW ADDING MORE
            self._wait_for_queue_space()
            if not self.closed:
                self.queue.extend(values)
            Log.note("{{name}} has {{num}} items", name=self.name, num=len(self.queue))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.add(THREAD_STOP)
        if isinstance(exc_val, BaseException):
            self.thread.please_stop.go()
        self.thread.join()

    def stop(self):
        self.add(THREAD_STOP)
        self.thread.join()



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

import mo_json
from mo_dots import Data, wrap
from mo_files import File
from mo_logs import Log
from mo_logs.exceptions import suppress_exception
from mo_math.randoms import Random
from mo_threads import Lock, Signal, THREAD_STOP

DEBUG = True


class PersistentQueue(object):
    """
    THREAD-SAFE, PERSISTENT QUEUE

    CAN HANDLE MANY PRODUCERS, BUT THE pop(), commit() IDIOM CAN HANDLE ONLY
    ONE CONSUMER.

    IT IS IMPORTANT YOU commit() or close(), OTHERWISE NOTHING COMES OFF THE QUEUE
    """

    def __init__(self, _file):
        """
        file - USES FILE FOR PERSISTENCE
        """
        self.file = File.new_instance(_file)
        self.lock = Lock("lock for persistent queue using file " + self.file.name)
        self.please_stop = Signal()
        self.db = Data()
        self.pending = []

        if self.file.exists:
            for line in self.file:
                with suppress_exception:
                    delta = mo_json.json2value(line)
                    apply_delta(self.db, delta)
            if self.db.status.start == None:  # HAPPENS WHEN ONLY ADDED TO QUEUE, THEN CRASH
                self.db.status.start = 0
            self.start = self.db.status.start

            # SCRUB LOST VALUES
            lost = 0
            for k in self.db.keys():
                with suppress_exception:
                    if k!="status" and int(k) < self.start:
                        self.db[k] = None
                        lost += 1
                  # HAPPENS FOR self.db.status, BUT MAYBE OTHER PROPERTIES TOO
            if lost:
                Log.warning("queue file had {{num}} items lost",  num= lost)

            if DEBUG:
                Log.note("Persistent queue {{name}} found with {{num}} items", name=self.file.abspath, num=len(self))
        else:
            self.db.status = Data(
                start=0,
                end=0
            )
            self.start = self.db.status.start
            if DEBUG:
                Log.note("New persistent queue {{name}}", name=self.file.abspath)

    def _add_pending(self, delta):
        delta = wrap(delta)
        self.pending.append(delta)

    def _apply_pending(self):
        for delta in self.pending:
            apply_delta(self.db, delta)
        self.pending = []


    def __iter__(self):
        """
        BLOCKING ITERATOR
        """
        while not self.please_stop:
            try:
                value = self.pop()
                if value is not THREAD_STOP:
                    yield value
            except Exception as e:
                Log.warning("Tell me about what happened here", cause=e)
        if DEBUG:
            Log.note("queue iterator is done")

    def add(self, value):
        with self.lock:
            if self.closed:
                Log.error("Queue is closed")

            if value is THREAD_STOP:
                if DEBUG:
                    Log.note("Stop is seen in persistent queue")
                self.please_stop.go()
                return

            self._add_pending({"add": {str(self.db.status.end): value}})
            self.db.status.end += 1
            self._add_pending({"add": {"status.end": self.db.status.end}})
            self._commit()
        return self

    def __len__(self):
        with self.lock:
            return self.db.status.end - self.start

    def __getitem__(self, item):
        return self.db[str(item + self.start)]

    def pop(self, timeout=None):
        """
        :param timeout: OPTIONAL DURATION
        :return: None, IF timeout PASSES
        """
        with self.lock:
            while not self.please_stop:
                if self.db.status.end > self.start:
                    value = self.db[str(self.start)]
                    self.start += 1
                    return value

                if timeout is not None:
                    with suppress_exception:
                        self.lock.wait(timeout=timeout)
                        if self.db.status.end <= self.start:
                            return None
                else:
                    self.lock.wait()

            if DEBUG:
                Log.note("persistent queue already stopped")
            return THREAD_STOP

    def pop_all(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        with self.lock:
            if self.please_stop:
                return [THREAD_STOP]
            if self.db.status.end == self.start:
                return []

            output = []
            for i in range(self.start, self.db.status.end):
                output.append(self.db[str(i)])

            self.start = self.db.status.end
            return output

    def rollback(self):
        with self.lock:
            if self.closed:
                return
            self.start = self.db.status.start
            self.pending = []

    def commit(self):
        with self.lock:
            if self.closed:
                Log.error("Queue is closed, commit not allowed")

            try:
                self._add_pending({"add": {"status.start": self.start}})
                for i in range(self.db.status.start, self.start):
                    self._add_pending({"remove": str(i)})

                if self.db.status.end - self.start < 10 or Random.range(0, 1000) == 0:  # FORCE RE-WRITE TO LIMIT FILE SIZE
                    # SIMPLY RE-WRITE FILE
                    if DEBUG:
                        Log.note("Re-write {{num_keys}} keys to persistent queue", num_keys=self.db.status.end - self.start)

                        for k in self.db.keys():
                            if k == "status" or int(k) >= self.db.status.start:
                                continue
                            Log.error("Not expecting {{key}}", key=k)
                    self._commit()
                    self.file.write(mo_json.value2json({"add": self.db}) + "\n")
                else:
                    self._commit()
            except Exception as e:
                raise e

    def _commit(self):
        self.file.append("\n".join(mo_json.value2json(p) for p in self.pending))
        self._apply_pending()

    def close(self):
        self.please_stop.go()
        with self.lock:
            if self.db is None:
                return

            self.add(THREAD_STOP)

            if self.db.status.end == self.start:
                if DEBUG:
                    Log.note("persistent queue clear and closed")
                self.file.delete()
            else:
                if DEBUG:
                    Log.note("persistent queue closed with {{num}} items left", num=len(self))
                try:
                    self._add_pending({"add": {"status.start": self.start}})
                    for i in range(self.db.status.start, self.start):
                        self._add_pending({"remove": str(i)})
                    self.file.write(mo_json.value2json({"add": self.db}) + "\n" + ("\n".join(mo_json.value2json(p) for p in self.pending)) + "\n")
                    self._apply_pending()
                except Exception as e:
                    raise e
            self.db = None

    @property
    def closed(self):
        with self.lock:
            return self.db is None


def apply_delta(value, delta):
    if delta.add:
        for k, v in delta.add.items():
            value[k] = v
    elif delta.remove:
        value[delta.remove] = None

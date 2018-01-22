# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from collections import namedtuple, deque
from mo_logs import Log


Step = namedtuple("Step", ["parent", "node"])


class Path(list):
    """
    USES Steps TO DEFINE A LIST
    Steps POINT TO parent, SO THIS CLASS HANDLES THE REVERSE NATURE
    """
    def __init__(self, last_step):
        self.last = last_step
        self.list = None

    def _build_list(self):
        output = deque()
        s = self.last
        while s:
            output.appendleft(s.node)
            s = s.parent
        self.list = list(output)

    def __getitem__(self, index):
        if index < 0:
            return None

        if not self.list:
            self._build_list()

        if index>=len(self.list):
            return None
        return self.list[index]

    def __setitem__(self, i, y):
        if not self.list:
            self._build_list()
        self.list[i]=y

    def __iter__(self):
        if not self.list:
            self._build_list()
        return self.list.__iter__()

    def __contains__(self, item):
        if not self.list:
            self._build_list()
        return item in self.list

    def append(self, val):
        Log.error("not implemented")

    def __str__(self):
        Log.error("not implemented")

    def __len__(self):
        if not self.list:
            self._build_list()
        return len(self.list)

    def __getslice__(self, i, j):
        Log.error("slicing is broken in Python 2.7: a[i:j] == a[i+len(a), j] sometimes.  Use [start:stop:step]")

    def copy(self):
        if not self.list:
            self._build_list()
        return self.list.copy()

    def remove(self, x):
        Log.error("not implemented")

    def extend(self, values):
        Log.error("not implemented")

    def pop(self):
        Log.error("not implemented")

    def right(self, num=None):
        """
        WITH SLICES BEING FLAT, WE NEED A SIMPLE WAY TO SLICE FROM THE RIGHT [-num:]
        """
        if num == None:
            return self.last.node
        if num <= 0:
            return []

        if not self.list:
            self._build_list()
        return self.list[-num:]

    def not_right(self, num):
        """
        WITH SLICES BEING FLAT, WE NEED A SIMPLE WAY TO SLICE FROM THE LEFT [:-num:]
        """
        if not self.list:
            self._build_list()

        if num == None:
            return self.list[:-1:]
        if num <= 0:
            return []

        return self.list[:-num:]

    def last(self):
        """
        RETURN LAST ELEMENT IN FlatList [-1]
        """
        return self.last.node


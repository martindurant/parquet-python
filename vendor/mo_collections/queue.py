
# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
from collections import deque


class Queue(object):
    """
    A SET WITH ADDED ORDER MAINTAINED

    +------------+---------+----------+
    | Uniqueness | Ordered | Type     |
    +------------+---------+----------+
    |     Yes    |   Yes   | Queue    |
    |     Yes    |   No    | Set      |
    |     No     |   Yes   | List     |
    |     No     |   No    | Multiset |
    +------------+---------+----------+
    """
    def __init__(self):
        self.set = set()
        self.list = deque()

    def __nonzero__(self):
        return len(self.list) > 0

    def __contains__(self, value):
        return value in self.set

    def __len__(self):
        return self.list.__len__()

    def __iter__(self):
        return iter(self.list)

    def __rsub__(self, other):
        if isinstance(other, set):
            return other - self.set
        return set(o for o in other if o not in self.set)

    def __data__(self):
        return list(self.list)

    def add(self, value):
        if value in self.set:
            return self
        self.set.add(value)
        self.list.append(value)

    def push(self, value):
        if value in self.set:
            self.list.remove(value)
        else:
            self.set.add(value)

        self.list.appendleft(value)

    def extend(self, values):
        for v in values:
            self.add(v)

    def pop(self):
        if len(self.list) == 0:
            return None

        output = self.list.popleft()
        self.set.remove(output)
        return output

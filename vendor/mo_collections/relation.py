# encoding: utf-8
#
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


class Relation_usingList(object):
    def __init__(self):
        self.all=set()

    def len(self):
        return len(self.all)

    def add(self, key, value):
        test = (key, value)
        if test not in self.all:
            self.all.add(test)

    def testAndAdd(self, key, value):
        """
        RETURN TRUE IF THIS RELATION IS NET-NEW
        """
        test = (key, value)
        if test not in self.all:
            self.all.add(test)
            return True
        return False

    def extend(self, key, values):
        for v in values:
            self.add(key, v)

    def __getitem__(self, key):
        """
        RETURN AN ARRAY OF OBJECTS THAT key MAPS TO
        """
        return [v for k, v in self.all if k == key]


class Relation(object):
    def __init__(self):
        self.map = dict()

    def len(self):
        return sum(len(v) for k, v in self.map.items() if v)

    def add(self, key, value):
        to = self.map.get(key)
        if to is None:
            to = set()
            self.map[key] = to
        to.add(value)

    def testAndAdd(self, key, value):
        """
        RETURN TRUE IF THIS RELATION IS NET-NEW
        """
        to = self.map.get(key)
        if to is None:
            to = set()
            self.map[key] = to
            to.add(value)
            return True

        if value in to:
            return False
        to.add(value)
        return True

    def extend(self, key, values):
        to = self.map.get(key)
        if not to:
            to = set(values)
            self.map[key] = to
            return

        to.update(values)

    def __getitem__(self, key):
        """
        RETURN AN ARRAY OF OBJECTS THAT key MAPS TO
        """
        o = self.map.get(key)
        if not o:
            return set()
        return o

    def domain(self):
        return self.map.keys()



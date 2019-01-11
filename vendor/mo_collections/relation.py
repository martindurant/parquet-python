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
from mo_logs import Log


class Relation_usingList(object):
    def __init__(self):
        self.all = set()

    def len(self):
        return len(self.all)

    def testAndAdd(self, key, value):
        """
        RETURN TRUE IF THIS RELATION IS NET-NEW
        """
        test = (key, value)
        output = test not in self.all
        self.all.add(test)
        return output

    def extend(self, key, values):
        for v in values:
            self[key] = v

    def __getitem__(self, key):
        """
        USE THIS IF YOU ARE CONFIDENT THIS IS A MANY-TO-ONE MAPPING
        RETURN THE SINGLE CO-DOMAIN OBJECT THIS key MAPS TO
        """
        output = [v for k, v in self.all if k == key]
        if not output:
            return None
        elif len(output) == 1:
            return output[0]
        else:
            Log.error("Not allowed")

    def __setitem__(self, key, value):
        self.all.add((key, value))

    def get_domain(self, value):
        """
        RETURN domain FOR GIVEN CODOMAIN
        :param value:
        :return:
        """
        return [k for k, v in self.all if v == value]

    def get_codomain(self, key):
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

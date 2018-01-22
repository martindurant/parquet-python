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



class Multiset(object):
    """
    Multiset IS ONE MEMBER IN A FAMILY OF USEFUL CONTAINERS

    +------------+---------+----------+
    | Uniqueness | Ordered | Type     |
    +------------+---------+----------+
    |     Yes    |   Yes   | Queue    |
    |     Yes    |   No    | Set      |
    |     No     |   Yes   | List     |
    |     No     |   No    | Multiset |
    +------------+---------+----------+
    """
    def __new__(cls, list=None, key_field=None, count_field=None, allow_negative=False):
        try:
            if allow_negative:
                return _NegMultiset(list, key_field, count_field)
            else:
                return _Multiset(list, key_field, count_field)
        except Exception as e:
            from mo_logs import Log

            Log.error("Not expected", e)

    def add(self, value):
        raise NotImplementedError

    def extend(self, values):
        raise NotImplementedError

    def remove(self, value):
        raise NotImplementedError


class _Multiset(Multiset):

    def __new__(cls, *args):
        return object.__new__(cls, *args)


    def __init__(self, list=None, key_field=None, count_field=None, **kwargs):
        if not key_field and not count_field:
            self.dic = dict()
            if list:
                for i in list:
                    self.add(i)
            return
        else:
            self.dic = {i[key_field]: i[count_field] for i in list}


    def __iter__(self):
        for k, m in self.dic.items():
            for i in range(m):
                yield k


    def items(self):
        return self.dic.items()

    def keys(self):
        return self.dic.keys()

    def add(self, value):
        if value in self.dic:
            self.dic[value] += 1
        else:
            self.dic[value] = 1
        return self

    def extend(self, values):
        for v in values:
            self.add(v)

    def remove(self, value):
        if value not in self.dic:
            from mo_logs import Log

            Log.error("{{value}} is not in multiset",  value= value)
        self._remove(value)

    def copy(self):
        output = _Multiset()
        output.dic = self.dic.copy()
        return output


    def _remove(self, value):
        count = self.dic.get(value)
        if count == None:
            return

        count -= 1
        if count == 0:
            del (self.dic[value])
        else:
            self.dic[value] = count

    def __sub__(self, other):
        output = self.copy()
        for o in other:
            output._remove(o)
        return output

    def __add__(self, other):
        output = self.copy()

        if isinstance(other, Multiset):
            for k, c in other.dic.items():
                output.dic[k] = output.dic.get(k, 0) + c
        else:
            for o in other:
                output.add(o)
        return output

    def __set__(self, other):
        return set(self.dic.keys())

    def __len__(self):
        return sum(self.dic.values())

    def __nonzero__(self):
        if self.dic:
            return True
        return False


    def count(self, value):
        if value in self.dic:
            return self.dic[value]
        else:
            return 0


class _NegMultiset(Multiset):
    def __new__(cls, *args, **kwargs):
            return object.__new__(cls, *args, **kwargs)

    def __init__(self, list=None, key_field=None, count_field=None, **kwargs):
        if not key_field and not count_field:
            self.dic = dict()
            if list:
                for i in list:
                    self.add(i)
            return
        else:
            self.dic = {i[key_field]: i[count_field] for i in list}


    # def __iter__(self):
    #     for k, m in self.dic.items():
    #         for i in range(m):
    #             yield k


    def items(self):
        return self.dic.items()

    def keys(self):
        return self.dic.keys()

    def add(self, value, amount=None):
        count = self.dic.get(value)
        if amount == None:
            amount = 1
        elif amount == 0:
            return self

        if not count:
            self.dic[value] = amount
        elif count == -amount:
            del (self.dic[value])
        else:
            self.dic[value] = count + amount

        return self

    def extend(self, values):
        for v in values:
            self.add(v)

    def remove(self, value):
        return self.add(value, -1)


    def copy(self):
        output = _NegMultiset()
        output.dic = self.dic.copy()
        return output


    def __add__(self, other):
        output = self.copy()

        if isinstance(other, _NegMultiset):
            for k, c in other.dic.items():
                output.add(k, c)
        else:
            for o in other:
                output.add(o)

        return output

    def __sub__(self, other):
        if not other:
            return self

        output = self.copy()
        for o in other:
            output.remove(o)
        return output

    def __set__(self, other):
        return set(self.dic.keys())

    def __len__(self):
        return sum(self.dic.values())

    def __nonzero__(self):
        if self.dic:
            return True
        return False


    def count(self, value):
        if value in self.dic:
            return self.dic[value]
        else:
            return 0

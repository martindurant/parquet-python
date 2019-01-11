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

from copy import copy
import functools

from mo_collections.multiset import Multiset
from mo_dots import FlatList
from mo_logs import Log
import mo_math
from mo_math import MIN, stats
from mo_math.stats import ZeroMoment, ZeroMoment2Stats


# A VARIETY OF SLIDING WINDOW FUNCTIONS


class AggregationFunction(object):
    def __init__(self):
        """
        RETURN A ZERO-STATE AGGREGATE
        """
        raise NotImplementedError

    def add(self, value):
        """
        ADD value TO AGGREGATE
        """
        raise NotImplementedError


    def merge(self, agg):
        """
        ADD TWO AGGREGATES TOGETHER
        """
        raise NotImplementedError

    def end(self):
        """
        RETURN AGGREGATE
        """
        raise NotImplementedError


class Exists(AggregationFunction):
    def __init__(self):
        object.__init__(self)
        self.total = False

    def add(self, value):
        if value == None:
            return
        self.total = True

    def merge(self, agg):
        if agg.total:
            self.total = True

    def end(self):
        return self.total


class One(AggregationFunction):
    """
    EXPECTING ONLY ONE VALUE OVER THE RESULT SET
    """
    def __init__(self, **kwargs):
        object.__init__(self)
        self.value = None

    def add(self, value):
        if value == None:
            return
        if self.value is None:
            self.value = value
            return
        if value != self.value:
            Log.error("Expecting value to match: {{expecting}}, {{instead}}",  expecting= self.value,  instead= value)

    def merge(self, agg):
        if self.value is None and agg.value is not None:
            self.value = agg.value
        elif self.value is not None:
            if self.value != agg.value:
                Log.error("Expecting value to match: {{expecting}}, {{instead}}",  expecting= self.value,  instead= agg.value)

    def end(self):
        return self.value


class WindowFunction(AggregationFunction):
    def __init__(self):
        """
        RETURN A ZERO-STATE AGGREGATE
        """
        raise NotImplementedError


    def sub(self, value):
        """
        REMOVE value FROM AGGREGATE
        """
        raise NotImplementedError


def Stats(**kwargs):
    if not kwargs:
        return _SimpleStats
    else:
        return functools.partial(_Stats, *[], **kwargs)


class _Stats(WindowFunction):
    """
    TRACK STATS, BUT IGNORE OUTLIERS
    """

    def __init__(self, middle=None, *args, **kwargs):
        object.__init__(self)
        self.middle = middle
        self.samples = FlatList()

    def add(self, value):
        if value == None:
            return
        self.samples.append(value)

    def sub(self, value):
        if value == None:
            return
        self.samples.remove(value)

    def merge(self, agg):
        Log.error("Do not know how to handle")

    def end(self):
        ignore = mo_math.ceiling(len(self.samples) * (1 - self.middle) / 2)
        if ignore * 2 >= len(self.samples):
            return stats.Stats()
        output = stats.Stats(samples=sorted(self.samples)[ignore:len(self.samples) - ignore:])
        output.samples = list(self.samples)
        return output


class _SimpleStats(WindowFunction):
    """
    AGGREGATE Stats OBJECTS, NOT JUST VALUES
    """

    def __init__(self, **kwargs):
        object.__init__(self)
        self.total = ZeroMoment(0, 0, 0)

    def add(self, value):
        if value == None:
            return
        self.total += ZeroMoment.new_instance([value])

    def sub(self, value):
        if value == None:
            return
        self.total -= ZeroMoment.new_instance([value])

    def merge(self, agg):
        self.total += agg.total

    def end(self):
        return ZeroMoment2Stats(self.total)


class Min(WindowFunction):
    def __init__(self, **kwargs):
        object.__init__(self)
        self.total = Multiset()


    def add(self, value):
        if value == None:

            return
        self.total.add(value)

    def sub(self, value):
        if value == None:
            return
        self.total.remove(value)

    def end(self):
        return MIN(self.total)


# class Max(WindowFunction):
#     def __init__(self, **kwargs):
#         object.__init__(self)
#         self.total = Multiset()
#
#
#     def add(self, value):
#         if value == None:
#             return
#         self.total.add(value)
#
#     def sub(self, value):
#         if value == None:
#             return
#         self.total.remove(value)
#
#     def end(self):
#         return MAX(self.total.dic.keys())


class Max(WindowFunction):
    def __init__(self, **kwargs):
        object.__init__(self)
        self.max = None


    def add(self, value):
        self.max = mo_math.MAX([self.max, value])

    def sub(self, value):
        raise NotImplementedError()

    def end(self):
        return self.max


class Count(WindowFunction):
    def __init__(self, **kwargs):
        object.__init__(self)
        self.total = 0


    def add(self, value):
        if value == None:
            return
        self.total += 1

    def sub(self, value):
        if value == None:
            return
        self.total -= 1

    def end(self):
        return self.total


class Sum(WindowFunction):
    def __init__(self, **kwargs):
        object.__init__(self)
        self.total = 0


    def add(self, value):
        if value == None:
            return
        self.total += value

    def sub(self, value):
        if value == None:
            return
        self.total -= value

    def end(self):
        return self.total


class Percentile(WindowFunction):
    def __init__(self, percentile, *args, **kwargs):
        """
        USE num_records TO MINIMIZE MEMORY CONSUPTION
        """
        object.__init__(self)
        self.percentile = percentile
        self.total = []


    def add(self, value):
        if value == None:
            return
        self.total.append(value)

    def sub(self, value):
        if value == None:
            return
        try:
            i = self.total.index(value)
            self.total = self.total[:i] + self.total[i+1:]
        except Exception as e:
            Log.error("Problem with window function", e)

    def end(self):
        return stats.percentile(self.total, self.percentile)


class List(WindowFunction):
    def __init__(self, **kwargs):
        object.__init__(self)
        self.agg = []

    def add(self, value):
        self.agg.append(value)

    def sub(self, value):
        if value != self.agg[0]:
            Log.error("Not a sliding window")
        self.agg = self.agg[1:]

    def end(self):
        return copy(self.agg)


name2accumulator = {
    "count": Count,
    "sum": Sum,
    "exists": Exists,
    "max": Max,
    "maximum": Max,
    "list": List,
    "min": Min,
    "minimum": Min,
    "percentile": Percentile,
    "one": One
}

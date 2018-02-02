# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with self file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import itertools
from collections import Mapping
from numbers import Number

from mo_future import text_type

from jx_base.expressions import jx_expression
from mo_collections.unique_index import UniqueIndex
from mo_dots import coalesce, Data, set_default, Null, listwrap
from mo_dots import wrap
from mo_dots.lists import FlatList
from mo_logs import Log
from mo_math import MAX, MIN
from mo_times.dates import Date
from mo_times.durations import Duration

ALGEBRAIC = {"time", "duration", "numeric", "count", "datetime"}  # DOMAINS THAT HAVE ALGEBRAIC OPERATIONS DEFINED
KNOWN = {"set", "boolean", "duration", "time", "numeric"}  # DOMAINS THAT HAVE A KNOWN NUMBER FOR PARTS AT QUERY TIME
PARTITION = {"uid", "set", "boolean"}  # DIMENSIONS WITH CLEAR PARTS


class Domain(object):
    __slots__ = ["name", "type", "value", "key", "label", "end", "isFacet", "where", "dimension", "primitive", "limit"]

    def __new__(cls, **desc):
        if cls == Domain:
            type_name = desc.get("type")
            try:
                return name_to_type[type_name](**desc)
            except Exception as e:
                Log.error("Problem with {\"domain\":{\"type\":{{type|quote}}}}", type=type_name, cause=e)
        else:
            return object.__new__(cls)

    def __init__(self, **desc):
        desc = wrap(desc)
        self._set_slots_to_null(self.__class__)
        set_default(self, desc)
        self.name = coalesce(desc.name, desc.type)
        self.isFacet = coalesce(desc.isFacet, False)
        self.dimension = Null
        self.limit = desc.limit

    def _set_slots_to_null(self, cls):
        """
        WHY ARE SLOTS NOT ACCESIBLE UNTIL WE ASSIGN TO THEM?
        """
        if hasattr(cls, "__slots__"):
            for s in cls.__slots__:
                self.__setattr__(s, Null)
        for b in cls.__bases__:
            self._set_slots_to_null(b)


    def __copy__(self):
        return self.__class__(**self.__data__())

    def copy(self):
        return self.__class__(**self.__data__())

    def __data__(self):
        return wrap({
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "key": self.key,
            "isFacet": self.isFacet,
            "where": self.where,
            "dimension": self.dimension
        })

    @property
    def __all_slots__(self):
        return self._all_slots(self.__class__)

    def _all_slots(self, cls):
        output = set(getattr(cls, '__slots__', []))
        for b in cls.__bases__:
            output |= self._all_slots(b)
        return output

    def getDomain(self):
        Log.error("Not implemented")

    def verify_attributes_not_null(self, attribute_names):
        for name in attribute_names:
            if getattr(self, name) == None:
                Log.error('{{type}} domain expects a {{name|quote}} parameter', type=self.type, name=name)


class UnitDomain(Domain):
    """
    REPRESENT A ZERO-DIMENSIONAL EDGE
    """
    def compare(self, a, b):
        return 0

    def getCanonicalPart(self, part):
        return 0

    def getPartByKey(self, key):
        return 0

    def getKey(self, part):
        return 0

    def getEnd(self, value):
        return 0


class ValueDomain(Domain):
    __slots__ = ["NULL"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        self.NULL = None

    def compare(self, a, b):
        return value_compare(a, b)

    def getCanonicalPart(self, part):
        return part

    def getPartByKey(self, key):
        return key

    def getKey(self, part):
        return part

    def getEnd(self, value):
        return value


class DefaultDomain(Domain):
    """
    DOMAIN IS A LIST OF OBJECTS, EACH WITH A value PROPERTY
    """

    __slots__ = ["NULL", "partitions", "map", "limit", "sort"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)

        self.NULL = Null
        self.partitions = FlatList()
        self.map = dict()
        self.map[None] = self.NULL
        self.limit = desc.get('limit')
        self.sort = 1

    def compare(self, a, b):
        return value_compare(a.value, b.value)

    def getCanonicalPart(self, part):
        return self.getPartByKey(part.value)

    def getPartByKey(self, key):
        canonical = self.map.get(key)
        if canonical:
            return canonical

        canonical = Data(name=key, value=key)

        self.partitions.append(canonical)
        self.map[key] = canonical
        return canonical

    # def getIndexByKey(self, key):
    #     return self.map.get(key).dataIndex;

    def getKey(self, part):
        return part.value

    def getEnd(self, part):
        return part.value

    def getLabel(self, part):
        return part.value

    def __data__(self):
        output = Domain.__data__(self)
        output.partitions = self.partitions
        output.limit = self.limit
        return output


class SimpleSetDomain(Domain):
    """
    DOMAIN IS A LIST OF OBJECTS, EACH WITH A value PROPERTY
    """

    __slots__ = ["NULL", "partitions", "map", "order"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        desc = wrap(desc)

        self.type = "set"
        self.order = {}
        self.NULL = Null
        self.partitions = FlatList()
        self.primitive = True  # True IF DOMAIN IS A PRIMITIVE VALUE SET

        if isinstance(self.key, set):
            Log.error("problem")

        if not desc.key and (len(desc.partitions)==0 or isinstance(desc.partitions[0], (text_type, Number, tuple))):
            # ASSUME PARTS ARE STRINGS, CONVERT TO REAL PART OBJECTS
            self.key = "value"
            self.map = {}
            self.order[None] = len(desc.partitions)
            for i, p in enumerate(desc.partitions):
                part = {"name": p, "value": p, "dataIndex": i}
                self.partitions.append(part)
                self.map[p] = part
                self.order[p] = i
                if isinstance(p, (int, float)):
                    text_part = text_type(float(p))  # ES CAN NOT HANDLE NUMERIC PARTS
                    self.map[text_part] = part
                    self.order[text_part] = i
            self.label = coalesce(self.label, "name")
            self.primitive = True
            return

        if desc.partitions and desc.dimension.fields and len(desc.dimension.fields) > 1:
            self.key = desc.key
            self.map = UniqueIndex(keys=desc.dimension.fields)
        elif desc.partitions and isinstance(desc.key, (list, set)):
            # TODO: desc.key CAN BE MUCH LIKE A SELECT, WHICH UniqueIndex CAN NOT HANDLE
            self.key = desc.key
            self.map = UniqueIndex(keys=desc.key)
        elif desc.partitions and isinstance(desc.partitions[0][desc.key], Mapping):
            self.key = desc.key
            self.map = UniqueIndex(keys=desc.key)
            # self.key = UNION(set(d[desc.key].keys()) for d in desc.partitions)
            # self.map = UniqueIndex(keys=self.key)
        elif len(desc.partitions) == 0:
            # CREATE AN EMPTY DOMAIN
            self.key = "value"
            self.map = {}
            self.order[None] = 0
            self.label = coalesce(self.label, "name")
            return
        elif desc.key == None:
            if desc.partitions and all(desc.partitions.where) or all(desc.partitions.esfilter):
                if not all(desc.partitions.name):
                    Log.error("Expecting all partitions to have a name")
                self.key = "name"
                self.map = dict()
                self.map[None] = self.NULL
                self.order[None] = len(desc.partitions)
                for i, p in enumerate(desc.partitions):
                    self.partitions.append({
                        "where": jx_expression(coalesce(p.where, p.esfilter)),
                        "name": p.name,
                        "dataIndex": i
                    })
                    self.map[p.name] = p
                    self.order[p.name] = i
                return
            elif desc.partitions and len(set(desc.partitions.value)-{None}) == len(desc.partitions):
                # TRY A COMMON KEY CALLED "value".  IT APPEARS UNIQUE
                self.key = "value"
                self.map = dict()
                self.map[None] = self.NULL
                self.order[None] = len(desc.partitions)
                for i, p in enumerate(desc.partitions):
                    self.map[p[self.key]] = p
                    self.order[p[self.key]] = i
                self.primitive = False
            else:
                Log.error("Domains must have keys, or partitions")
        elif self.key:
            self.key = desc.key
            self.map = dict()
            self.map[None] = self.NULL
            self.order[None] = len(desc.partitions)
            for i, p in enumerate(desc.partitions):
                self.map[p[self.key]] = p
                self.order[p[self.key]] = i
            self.primitive = False
        else:
            Log.error("Can not hanldle")

        self.label = coalesce(self.label, "name")

        if hasattr(desc.partitions, "__iter__"):
            self.partitions = wrap(list(desc.partitions))
        else:
            Log.error("expecting a list of partitions")

    def compare(self, a, b):
        return value_compare(self.getKey(a), self.getKey(b))

    def getCanonicalPart(self, part):
        return self.getPartByKey(part.value)

    def getIndexByKey(self, key):
        try:
            output = self.order.get(key)
            if output is None:
                return len(self.partitions)
            return output
        except Exception as e:
            Log.error("problem", e)


    def getPartByKey(self, key):
        try:
            canonical = self.map.get(key)
            if not canonical:
                return self.NULL
            return canonical
        except Exception as e:
            Log.error("problem", e)

    def getPartByIndex(self, index):
        return self.partitions[index]

    def getKeyByIndex(self, index):
        if index < 0 or index >= len(self.partitions):
            return None
        return self.partitions[index][self.key]

    def getKey(self, part):
        return part[self.key]

    def getEnd(self, part):
        if self.value:
            return part[self.value]
        else:
            return part

    def getLabel(self, part):
        return part[self.label]

    def __data__(self):
        output = Domain.__data__(self)
        output.partitions = self.partitions
        return output


class SetDomain(Domain):
    __slots__ = ["NULL", "partitions", "map", "order"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        desc = wrap(desc)

        self.type = "set"
        self.order = {}
        self.NULL = Null
        self.partitions = FlatList()

        if isinstance(self.key, set):
            Log.error("problem")

        if isinstance(desc.partitions[0], (int, float, text_type)):
            # ASSMUE PARTS ARE STRINGS, CONVERT TO REAL PART OBJECTS
            self.key = "value"
            self.order[None] = len(desc.partitions)
            for i, p in enumerate(desc.partitions):
                part = {"name": p, "value": p, "dataIndex": i}
                self.partitions.append(part)
                self.map[p] = part
                self.order[p] = i
        elif desc.partitions and desc.dimension.fields and len(desc.dimension.fields) > 1:
            self.key = desc.key
            self.map = UniqueIndex(keys=desc.dimension.fields)
        elif desc.partitions and isinstance(desc.key, (list, set)):
            # TODO: desc.key CAN BE MUCH LIKE A SELECT, WHICH UniqueIndex CAN NOT HANDLE
            self.key = desc.key
            self.map = UniqueIndex(keys=desc.key)
        elif desc.partitions and isinstance(desc.partitions[0][desc.key], Mapping):
            self.key = desc.key
            self.map = UniqueIndex(keys=desc.key)
            # self.key = UNION(set(d[desc.key].keys()) for d in desc.partitions)
            # self.map = UniqueIndex(keys=self.key)
        elif desc.key == None:
            Log.error("Domains must have keys")
        elif self.key:
            self.key = desc.key
            self.map = dict()
            self.map[None] = self.NULL
            self.order[None] = len(desc.partitions)
            for i, p in enumerate(desc.partitions):
                self.map[p[self.key]] = p
                self.order[p[self.key]] = i
        elif all(p.esfilter for p in self.partitions):
            # EVERY PART HAS AN esfilter DEFINED, SO USE THEM
            for i, p in enumerate(self.partitions):
                p.dataIndex = i

        else:
            Log.error("Can not hanldle")

        self.label = coalesce(self.label, "name")

    def compare(self, a, b):
        return value_compare(self.getKey(a), self.getKey(b))

    def getCanonicalPart(self, part):
        return self.getPartByKey(part.value)

    def getIndexByKey(self, key):
        try:
            output = self.order.get(key)
            if output is None:
                return len(self.partitions)
            return output
        except Exception as e:
            Log.error("problem", e)


    def getPartByKey(self, key):
        try:
            canonical = self.map.get(key, None)
            if not canonical:
                return self.NULL
            return canonical
        except Exception as e:
            Log.error("problem", e)

    def getKey(self, part):
        return part[self.key]

    def getKeyByIndex(self, index):
        return self.partitions[index][self.key]

    def getEnd(self, part):
        if self.value:
            return part[self.value]
        else:
            return part

    def getLabel(self, part):
        return part[self.label]

    def __data__(self):
        output = Domain.__data__(self)
        output.partitions = self.partitions
        return output


class TimeDomain(Domain):
    __slots__ = ["max", "min", "interval", "partitions", "NULL", "sort"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        self.type = "time"
        self.NULL = Null
        self.min = Date(self.min)
        self.max = Date(self.max)
        self.interval = Duration(self.interval)
        self.sort = Null

        if self.partitions:
            # IGNORE THE min, max, interval
            if not self.key:
                Log.error("Must have a key value")

            Log.error("not implemented yet")

            # VERIFY PARTITIONS DO NOT OVERLAP
            return

        self.verify_attributes_not_null(["min", "max", "interval"])
        self.key = "min"
        self.partitions = wrap([
            {"min": v, "max": v + self.interval, "dataIndex": i}
            for i, v in enumerate(Date.range(self.min, self.max, self.interval))
        ])

    def compare(self, a, b):
        return value_compare(a, b)

    def getCanonicalPart(self, part):
        return self.getPartByKey(part[self.key])

    def getIndexByKey(self, key):
        for p in self.partitions:
            if p.min <= key < p.max:
                return p.dataIndex
        return len(self.partitions)

    def getPartByKey(self, key):
        for p in self.partitions:
            if p.min <= key < p.max:
                return p
        return self.NULL

    def getKey(self, part):
        return part[self.key]

    def getKeyByIndex(self, index):
        return self.partitions[index][self.key]

    def __data__(self):
        output = Domain.__data__(self)

        output.partitions = self.partitions
        output.min = self.min
        output.max = self.max
        output.interval = self.interval
        return output


class DurationDomain(Domain):
    __slots__ = ["max", "min", "interval", "partitions", "NULL"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        self.type = "duration"
        self.NULL = Null
        self.min = Duration(self.min)
        self.max = Duration(self.max)
        self.interval = Duration(self.interval)

        if self.partitions:
            # IGNORE THE min, max, interval
            if not self.key:
                Log.error("Must have a key value")

            Log.error("not implemented yet")

            # VERIFY PARTITIONS DO NOT OVERLAP
            return
        elif not all([self.min, self.max, self.interval]):
            Log.error("Can not handle missing parameter")

        self.key = "min"
        self.partitions = wrap([{"min": v, "max": v + self.interval, "dataIndex":i} for i, v in enumerate(Duration.range(self.min, self.max, self.interval))])

    def compare(self, a, b):
        return value_compare(a, b)

    def getCanonicalPart(self, part):
        return self.getPartByKey(part[self.key])

    def getIndexByKey(self, key):
        for p in self.partitions:
            if p.min <= key < p.max:
                return p.dataIndex
        return len(self.partitions)

    def getPartByKey(self, key):
        for p in self.partitions:
            if p.min <= key < p.max:
                return p
        return self.NULL

    def getKey(self, part):
        return part[self.key]

    def getKeyByIndex(self, index):
        return self.partitions[index][self.key]

    def __data__(self):
        output = Domain.__data__(self)

        output.partitions = self.partitions
        output.min = self.min
        output.max = self.max
        output.interval = self.interval
        return output



class NumericDomain(Domain):
    __slots__ = ["max", "min"]

    def __new__(cls, **desc):
        if not desc.get('partitions') and not desc.get('interval'):
            return object.__new__(cls)
        else:
            return object.__new__(RangeDomain)

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        self.min = desc.get('min')
        self.max = desc.get('max')

    def compare(self, a, b):
        return value_compare(a, b)

    def getCanonicalPart(self, part):
        return part

    def getIndexByKey(self, key):
        return key

    def getPartByKey(self, key):
        if self.min!=None and key < self.min:
            return self.NULL
        if self.max!=None and key >= self.max:
            return self.NULL
        return key

    def getKey(self, part):
        return part

    def getKeyByIndex(self, index):
        return index

    def __data__(self):
        output = Domain.__data__(self)

        output.min = self.min
        output.max = self.max
        return output


class UniqueDomain(Domain):
    __slots__ = ()

    def compare(self, a, b):
        return value_compare(a, b)

    def getCanonicalPart(self, part):
        return part

    def getPartByKey(self, key):
        return key

    def getKey(self, part):
        return part

    def getEnd(self, value):
        return value


class RangeDomain(Domain):
    __slots__ = ["max", "min", "interval", "partitions", "NULL"]

    def __init__(self, **desc):
        Domain.__init__(self, **desc)
        self.type = "range"
        self.NULL = Null

        if self.partitions:
            # IGNORE THE min, max, interval
            if not self.key:
                Log.error("Must have a key value")

            parts = list(listwrap(self.partitions))
            for i, p in enumerate(parts):
                self.min = MIN([self.min, p.min])
                self.max = MAX([self.max, p.max])
                if p.dataIndex != None and p.dataIndex != i:
                    Log.error("Expecting `dataIndex` to agree with the order of the parts")
                if p[self.key] == None:
                    Log.error("Expecting all parts to have {{key}} as a property", key=self.key)
                p.dataIndex = i

            # VERIFY PARTITIONS DO NOT OVERLAP, HOLES ARE FINE
            for p, q in itertools.product(parts, parts):
                if p is not q and p.min <= q.min and q.min < p.max:
                    Log.error("partitions overlap!")

            self.partitions = parts
            return
        elif any([self.min == None, self.max == None, self.interval == None]):
            Log.error("Can not handle missing parameter")

        self.key = "min"
        self.partitions = wrap([{"min": v, "max": v + self.interval, "dataIndex": i} for i, v in enumerate(frange(self.min, self.max, self.interval))])

    def compare(self, a, b):
        return value_compare(a, b)

    def getCanonicalPart(self, part):
        return self.getPartByKey(part[self.key])

    def getIndexByKey(self, key):
        for p in self.partitions:
            if p.min <= key < p.max:
                return p.dataIndex
        return len(self.partitions)

    def getPartByKey(self, key):
        for p in self.partitions:
            if p.min <= key < p.max:
                return p
        return self.NULL

    def getKey(self, part):
        return part[self.key]

    def getKeyByIndex(self, index):
        return self.partitions[index][self.key]

    def __data__(self):
        output = Domain.__data__(self)

        output.partitions = self.partitions
        output.min = self.min
        output.max = self.max
        output.interval = self.interval
        return output


def frange(start, stop, step):
    # LIKE range(), BUT FOR FLOATS
    output = start
    while output < stop:
        yield output
        output += step


def value_compare(a, b):
    if a == None:
        if b == None:
            return 0
        return -1
    elif b == None:
        return 1

    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0


name_to_type = {
    "value": ValueDomain,
    "default": DefaultDomain,
    "set": SimpleSetDomain,
    "time": TimeDomain,
    "duration": DurationDomain,
    "range": NumericDomain,
    "uid": UniqueDomain,
    "numeric": NumericDomain
}


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

from collections import Mapping, Iterable, Set

from mo_dots import unwrap, tuplewrap, wrap
from mo_dots.objects import datawrap
from mo_future import PY2
from mo_logs import Log
from mo_logs.exceptions import suppress_exception

DEBUG = False


class UniqueIndex(Set, Mapping):
    """
    DEFINE A SET OF ATTRIBUTES THAT UNIQUELY IDENTIFIES EACH OBJECT IN A list.
    THIS ALLOWS set-LIKE COMPARISIONS (UNION, INTERSECTION, DIFFERENCE, ETC) WHILE
    STILL MAINTAINING list-LIKE FEATURES
    KEYS CAN BE DOT-DELIMITED PATHS TO DEEP INNER OBJECTS
    """

    def __init__(self, keys, data=None, fail_on_dup=True):
        self._data = {}
        self._keys = tuplewrap(keys)
        self.count = 0
        self.fail_on_dup = fail_on_dup
        if data:
            for d in data:
                self.add(d)

    def __getitem__(self, key):
        try:
            _key = value2key(self._keys, key)
            if len(self._keys) == 1 or len(_key) == len(self._keys):
                d = self._data.get(_key)
                return wrap(d)
            else:
                output = wrap([
                    d
                    for d in self._data.values()
                    if all(wrap(d)[k] == v for k, v in _key.items())
                ])
                return output
        except Exception as e:
            Log.error("something went wrong", e)

    def __setitem__(self, key, value):
        Log.error("Use add() to ad to an index")
        # try:
        #     key = value2key(self._keys, key)
        #     d = self._data.get(key)
        #     if d != None:
        #         Log.error("key already filled")
        #     self._data[key] = unwrap(value)
        #     self.count += 1
        #
        # except Exception as e:
        #     Log.error("something went wrong", e)

    def keys(self):
        return self._data.keys()

    def pop(self):
        output = self._data.iteritems().next()[1]
        self.remove(output)
        return wrap(output)

    def add(self, val):
        val = datawrap(val)
        key = value2key(self._keys, val)
        if key == None:
            Log.error("Expecting key to be not None")

        d = self._data.get(key)
        if d is None:
            self._data[key] = unwrap(val)
            self.count += 1
        elif d is not val:
            if self.fail_on_dup:
                Log.error("{{new|json}} with key {{key|json}} already filled with {{old|json}}", key=key, new=val, old=self[val])
            elif DEBUG:
                Log.warning("key {{key|json}} already filled\nExisting\n{{existing|json|indent}}\nValue\n{{value|json|indent}}",
                    key=key,
                    existing=d,
                    value=val
                )

    def extend(self, values):
        for v in values:
            self.add(v)

    def remove(self, val):
        key = value2key(self._keys, datawrap(val))
        if key == None:
            Log.error("Expecting key to not be None")

        d = self._data.get(key)
        if d is None:
            # ALREADY GONE
            return
        else:
            del self._data[key]
            self.count -= 1

    def __contains__(self, key):
        return self[key] != None

    if PY2:
        def __iter__(self):
            return (wrap(v) for v in self._data.itervalues())
    else:
        def __iter__(self):
            return (wrap(v) for v in self._data.values())

    def __sub__(self, other):
        output = UniqueIndex(self._keys, fail_on_dup=self.fail_on_dup)
        for v in self:
            if v not in other:
                output.add(v)
        return output

    def __and__(self, other):
        output = UniqueIndex(self._keys)
        for v in self:
            if v in other:
                output.add(v)
        return output

    def __or__(self, other):
        output = UniqueIndex(self._keys)
        for v in self:
            output.add(v)
        for v in other:
            with suppress_exception:
                output.add(v)
        return output

    def __ior__(self, other):
        for v in other:
            with suppress_exception:
                self.add(v)

        return self

    def __xor__(self, other):
        if not isinstance(other, Iterable):
            Log.error("Expecting other to be iterable")
        other = UniqueIndex(keys=self._keys, data=other, fail_on_dup=False)
        return (self-other) | (other-self)

    def __len__(self):
        if self.count == 0:
            for d in self:
                self.count += 1
        return self.count

    def subtract(self, other):
        return self.__sub__(other)

    def intersect(self, other):
        return self.__and__(other)


def value2key(keys, val):
    if len(keys) == 1:
        if isinstance(val, Mapping):
            return val[keys[0]]
        elif isinstance(val, (list, tuple)):
            return val[0]
        else:
            return val
    else:
        if isinstance(val, Mapping):
            return datawrap({k: val[k] for k in keys})
        elif isinstance(val, (list, tuple)):
            return datawrap(dict(zip(keys, val)))
        else:
            Log.error("do not know what to do here")

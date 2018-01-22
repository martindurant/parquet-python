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

from collections import Mapping
from copy import copy

from mo_dots import wrap, unwrap, tuplewrap, get_attr
from mo_logs import Log


class Index(object):
    """
    USING DATABASE TERMINOLOGY, THIS IS A NON-UNIQUE INDEX
    KEYS CAN BE DOT-DELIMITED PATHS TO DEEP INNER OBJECTS
    """

    def __init__(self, keys, data=None):
        self._data = {}
        self._keys = tuplewrap(keys)
        self.count = 0

        if data:
            for i, d in enumerate(data):
                self.add(d)

    def __getitem__(self, key):
        try:
            if isinstance(key, (list, tuple)) and len(key) < len(self._keys):
                # RETURN ANOTHER Index
                Log.error("not implemented")

            key = value2key(self._keys, key)
            return wrap(copy(self._data.get(key, [])))
        except Exception as e:
            Log.error("something went wrong", e)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def add(self, val):
        key = value2key(self._keys, val)
        e = self._data.get(key, [])
        self._data[key] = e
        e.append(unwrap(val))
        self.count += 1


    def __contains__(self, key):
        expected = True if self[key] else False
        testing = self._test_contains(key)

        if testing == expected:
            return testing
        else:
            Log.error("not expected")

    def _test_contains(self, key):
        try:
            if isinstance(key, (list, tuple)) and len(key) < len(self._keys):
                # RETURN ANOTHER Index
                length = len(key)
                key = value2key(self._keys[0:length:], key)
                d = self._data
                for k in key[:length]:
                    try:
                        d = d[k]
                    except Exception as e:
                        return False
                return True

            key = value2key(self._keys, key)
            d = self._data
            for k in key:
                try:
                    d = d[k]
                except Exception as e:
                    return False
            return True
        except Exception as e:
            Log.error("something went wrong", e)

    def keys(self):
        if len(self._keys) == 1:
            return (k[0] for k in self._data.keys())
        else:
            return self._data.keys()

    def items(self):
        if len(self._keys)==1:
            return ((k[0], d) for k,d in self._data.items())
        else:
            return self._data.items()

    def __nonzero__(self):
        if self._data.keys():
            return True
        else:
            return False

    def __iter__(self):
        def iter(data, depth):
            if depth == 0:
                for v in data:
                    yield wrap(v)
                return

            for v in data.values():
                for v1 in iter(v, depth - 1):
                    yield wrap(v1)

        return iter(self._data, len(self._keys))

    def __sub__(self, other):
        output = Index(self._keys)
        for v in self:
            if v not in other:
                output.add(v)
        return output

    def __and__(self, other):
        output = Index(self._keys)
        for v in self:
            if v in other:
                output.add(v)
        return output

    def __or__(self, other):
        output = Index(self._keys)
        for v in self:
            output.add(v)
        for v in other:
            output.add(v)
        return output

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
            return get_attr(val, keys[0]),
        elif isinstance(val, (list, tuple)):
            return val[0],
        return val,
    else:
        if isinstance(val, Mapping):
            return tuple(val[k] for k in keys)
        elif isinstance(val, (list, tuple)):
            return tuple(val)
        else:
            Log.error("do not know what to do here")

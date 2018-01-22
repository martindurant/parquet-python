# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

# REPLACE NUMPY ARRAY FUNCTIONS
# THIS CODE IS FASTER THAN NUMPY WHEN USING PYPY *AND* THE ARRAYS ARE SMALL

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from mo_logs import Log


def zeros(dim):
    if not isinstance(dim, tuple):
        return [0] * dim

    if len(dim) == 1:
        return [0.0] * dim[0]

    return [zeros(dim[1::]) for i in range(dim[0])]


def ones(dim):
    if not isinstance(dim, tuple):
        return [1.0] * dim

    if len(dim) == 1:
        return [1.0] * dim[0]

    return [zeros(dim[1::]) for _ in range(dim[0])]


def _apply(func):
    def output(value):
        if hasattr(value, "__iter__"):
            return [output(v) for v in value]
        else:
            return func(value)

    return lambda v: array(output(v))


def _reduce(func):
    def agg(values, axis, depth):
        if depth == axis:
            return func

        if hasattr(values[0], "__iter__"):
            return [func(v) for v in values]
        else:
            return func(values)

    def agg_all(values):
        if hasattr(values[0], "__iter__"):
            return func([agg_all(v) for v in values])
        else:
            return func(values)

    def output(v, axis=None):
        if axis is None:
            return array(agg_all(v))
        else:
            return array(agg(v, axis, 0))

    return output


def _binary_op(op):
    def output(a, b):
        if hasattr(a, "__iter__"):
            if hasattr(b, "___iter__"):
                return [output(ai, bi) for ai, bi in zip(a, b)]
            else:
                return [output(ai, b) for ai in a]
        else:
            if hasattr(b, "___iter__"):
                return [output(a, bi) for bi in b]
            else:
                return op(a, b)

    return lambda a, b: array(output(a, b))


g = globals()
MATH = g["math"].__dict__.copy()
for k, f in MATH.items():
    if hasattr(f, '__call__'):
        g[k] = _apply(f)

MORE_MATH = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "sub": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "mult": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "div": lambda a, b: a / b
}
for k, f in MORE_MATH.items():
    g[k] = _apply(f)

AGGS = {
    "min": min,
    "sum": sum,
    "max": max,
    "argmax": max,
    "argmin": min,
    "mean": lambda v: sum(v) / float(len(v)) if v else None,
    "var": lambda vs: sum([v ** 2 for v in vs]) - (sum(vs) / float(len(vs))) ** 2
}
for k, f in AGGS.items():  # AGGREGATION
    g[k] = _reduce(f)

IGNORE = [
    "__array__",
    "__array_interface__",
    "__array_struct__",
    "__coerce__"
]


def dot(a, b):
    Log.error("Not implemented yet")


def transpose(a):
    raise NotImplementedError


def seterr(*args, **kwargs):
    pass


def allclose(a, b):
    try:
        assertAlmostEqual(a, b)
        return True
    except Exception as e:
        return False


def array(value, dtype=None):
    if isinstance(value, _array):
        return value
    if hasattr(value, "__iter__"):
        return _array(value)
    return value


class _array:
    def __init__(self, value):
        self._value = value

    def __getattr__(self, item):
        if item in IGNORE:
            pass
        else:
            Log.error("operation {{op}} not found", op=item)

    def __iter__(self):
        return self._value.__iter__()

    def __index__(self, items):
        return [self[i] for i in items]

    def __nonzero__(self):
        return len(self._value) > 0

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return self._value.__getitem__(item)

        def _get_item(vals, items):
            if len(items) > 1:
                if isinstance(items[0], slice):
                    return [_get_item(val, items[1:]) for val in vals[items[0]]]
                else:
                    return _get_item(vals[items[0]], items[1:])
            else:
                return vals[items[0]]

        return array(_get_item(self._value, item))

    def __len__(self):
        return self._value.__len__()

    def __repr__(self):
        return self._value.__repr__()

    def __str__(self):
        return self._value.__str__()

    def tolist(self):
        return list(self._value)

    @property
    def shape(self):
        def _shape(val):
            if hasattr(val, "__iter__"):
                return (len(val),) + _shape(val[0])
            else:
                return ()

        return _shape(self._value)

    def astype(self, type):
        return self

    def __deepcopy__(self):
        pass


# ADD AGGREGATES TO CLASS
for k, f in AGGS.items():
    _array.__dict__[k] = _reduce(f)

# DEFINE THE OPERATORS ON CLASS
for k, op in MORE_MATH.items():
    _array.__dict__["__" + k + "__"] = lambda self, other: _binary_op(op, self._value, other)
    _array.__dict__["__r" + k + "__"] = lambda self, other: _binary_op(op, self._value, other)


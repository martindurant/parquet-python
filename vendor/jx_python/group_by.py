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

import math
import sys

from mo_dots import listwrap, Null, Data
from mo_future import text_type, binary_type
from mo_logs import Log

from jx_base.container import Container
from jx_base.expressions import jx_expression, Expression
from jx_python.expressions import jx_expression_to_function
from mo_collections.multiset import Multiset
from mo_dots.lists import FlatList
from mo_logs.exceptions import Except


def groupby(data, keys=None, size=None, min_size=None, max_size=None, contiguous=False):
    """
    :param data:
    :param keys:
    :param size:
    :param min_size:
    :param max_size:
    :param contiguous: MAINTAIN THE ORDER OF THE DATA, STARTING THE NEW GROUP WHEN THE SELECTOR CHANGES
    :return: return list of (keys, values) PAIRS, WHERE
                 keys IS IN LEAF FORM (FOR USE WITH {"eq": terms} OPERATOR
                 values IS GENERATOR OF ALL VALUE THAT MATCH keys
        contiguous -
    """
    if isinstance(data, Container):
        return data.groupby(keys)

    if size != None or min_size != None or max_size != None:
        if size != None:
            max_size = size
        return groupby_min_max_size(data, min_size=min_size, max_size=max_size)

    try:
        keys = listwrap(keys)
        if not contiguous:
            from jx_python import jx
            data = jx.sort(data, keys)

        if not data:
            return Null

        if any(isinstance(k, Expression) for k in keys):
            Log.error("can not handle expressions")
        else:
            accessor = jx_expression_to_function(jx_expression({"tuple": keys}))  # CAN RETURN Null, WHICH DOES NOT PLAY WELL WITH __cmp__

        def _output():
            start = 0
            prev = accessor(data[0])
            for i, d in enumerate(data):
                curr = accessor(d)
                if curr != prev:
                    group = {}
                    for k, gg in zip(keys, prev):
                        group[k] = gg
                    yield Data(group), data[start:i:]
                    start = i
                    prev = curr
            group = {}
            for k, gg in zip(keys, prev):
                group[k] = gg
            yield Data(group), data[start::]

        return _output()
    except Exception as e:
        Log.error("Problem grouping", cause=e)


def groupby_size(data, size):
    if hasattr(data, "next"):
        iterator = data
    elif hasattr(data, "__iter__"):
        iterator = data.__iter__()
    else:
        Log.error("do not know how to handle this type")

    done = FlatList()
    def more():
        output = FlatList()
        for i in range(size):
            try:
                output.append(iterator.next())
            except StopIteration:
                done.append(True)
                break
        return output

    # THIS IS LAZY
    i = 0
    while True:
        output = more()
        yield (i, output)
        if len(done) > 0:
            break
        i += 1


def groupby_Multiset(data, min_size, max_size):
    # GROUP multiset BASED ON POPULATION OF EACH KEY, TRYING TO STAY IN min/max LIMITS
    if min_size == None:
        min_size = 0

    total = 0
    i = 0
    g = list()
    for k, c in data.items():
        if total < min_size or total + c < max_size:
            total += c
            g.append(k)
        elif total < max_size:
            yield (i, g)
            i += 1
            total = c
            g = [k]

        if total >= max_size:
            Log.error("({{min}}, {{max}}) range is too strict given step of {{increment}}",
                min=min_size,
                max=max_size,
                increment=c
            )

    if g:
        yield (i, g)


def groupby_min_max_size(data, min_size=0, max_size=None, ):
    if max_size == None:
        max_size = sys.maxint

    if isinstance(data, (bytearray, text_type, binary_type, list)):
        def _iter():
            num = int(math.ceil(len(data)/max_size))
            for i in range(num):
                output = (i, data[i * max_size:i * max_size + max_size:])
                yield output

        return _iter()

    elif hasattr(data, "__iter__"):
        def _iter():
            g = 0
            out = FlatList()
            try:
                for i, d in enumerate(data):
                    out.append(d)
                    if (i + 1) % max_size == 0:
                        yield g, out
                        g += 1
                        out = FlatList()
                if out:
                    yield g, out
            except Exception as e:
                e = Except.wrap(e)
                if out:
                    # AT LEAST TRY TO RETURN WHAT HAS BEEN PROCESSED SO FAR
                    yield g, out
                Log.error("Problem inside jx.groupby", e)

        return _iter()
    elif not isinstance(data, Multiset):
        return groupby_size(data, max_size)
    else:
        return groupby_Multiset(data, min_size, max_size)


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
from collections import Mapping

import functools
from mo_math import MIN
from mo_logs import Log
from mo_dots import split_field, coalesce, Data, FlatList, wrap


class PartFlatList(list):
    """
    FlatList IS A RESULT OF FILTERING SETS OF TREES
    WE SAVED OURSELVES FROM COPYING ALL OBJECTS IN ALL PATHS OF ALL TREES,
    BUT WE ARE LEFT WITH THIS LIST OF TUPLES THAT POINT TO THE SAME
    """

    def __init__(self, path, data):
        """
        data IS A LIST OF TUPLES
        EACH TUPLE IS THE SEQUENCE OF OBJECTS FOUND ALONG A PATH IN A TREE
        IT IS EXPECTED len(data[i]) == len(path)+1 (data[i][0] IS THE ORIGINAL ROW OBJECT)
        """
        list.__init__(self)
        self.data = data
        self.path = path

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        """
        WE ARE NOW DOOMED TO COPY THE RECORDS (BECAUSE LISTS DOWN THE PATH ARE SPECIFIC ELEMENTS)
        """
        for d in self.data:
            r = d[-1]
            for i in range(len(self.path)):
                temp = dict(d[-i - 2])
                temp[self.path[-i - 1]] = r
                r = temp
            yield r

    def select(self, fields):
        if isinstance(fields, Mapping):
            fields=fields.value

        if isinstance(fields, text_type):
            # RETURN LIST OF VALUES
            if len(split_field(fields)) == 1:
                if self.path[0] == fields:
                    return [d[1] for d in self.data]
                else:
                    return [d[0][fields] for d in self.data]
            else:
                keys = split_field(fields)
                depth = coalesce(MIN([i for i, (k, p) in enumerate(zip(keys, self.path)) if k != p]), len(self.path))  # LENGTH OF COMMON PREFIX
                short_key = keys[depth:]

                output = FlatList()
                _select1((wrap(d[depth]) for d in self.data), short_key, 0, output)
                return output

        if isinstance(fields, list):
            output = FlatList()

            meta = []
            for f in fields:
                if hasattr(f.value, "__call__"):
                    meta.append((f.name, f.value))
                else:
                    meta.append((f.name, functools.partial(lambda v, d: d[v], f.value)))

            for row in self._values():
                agg = Data()
                for name, f in meta:
                    agg[name] = f(row)

                output.append(agg)

            return output

            # meta = []
            # for f in fields:
            #     keys = split_field(f.value)
            #     depth = coalesce(MIN([i for i, (k, p) in enumerate(zip(keys, self.path)) if k != p]), len(self.path))  # LENGTH OF COMMON PREFIX
            #     short_key = join_field(keys[depth:])
            #
            #     meta.append((f.name, depth, short_key))
            #
            # for row in self._data:
            #     agg = Data()
            #     for name, depth, short_key in meta:
            #         if short_key:
            #             agg[name] = row[depth][short_key]
            #         else:
            #             agg[name] = row[depth]
            #     output.append(agg)
            # return output

        Log.error("multiselect over FlatList not supported")

    def _values(self):
        temp = [[]] * len(self.path)
        for d in self.data:
            for i, p in enumerate(self.path):
                temp[i] = d[i][p]    # REMEMBER THE LIST THAT IS HERE
                d[i][p] = d[i + 1]   # REPLACE WITH INSTANCE
            yield d[0]               # DO THE WORK
            for i, p in enumerate(self.path):
                d[i][p] = temp[i]    # RETURN LIST BACK TO PLACE


def _select1(data, field, depth, output):
    """
    SELECT A SINGLE FIELD
    """
    for d in data:
        for i, f in enumerate(field[depth:]):
            d = d[f]
            if d == None:
                output.append(None)
                break
            elif isinstance(d, list):
                _select1(d, field, i + 1, output)
                break
        else:
            output.append(d)

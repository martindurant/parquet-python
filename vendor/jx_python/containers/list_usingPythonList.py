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
import itertools

import jx_base
from jx_base import Container
from jx_base.expressions import TRUE, Variable
from jx_base.language import is_expression, is_op
from jx_python.expressions import jx_expression_to_function
from jx_python.lists.aggs import is_aggs, list_aggs
from jx_python.meta import get_schema_from_list
from mo_collections import UniqueIndex
from mo_dots import Data, Null, is_data, is_list, listwrap, unwrap, unwraplist, wrap
from mo_future import first, sort_using_key
from mo_logs import Log
from mo_threads import Lock
from pyLibrary import convert


class ListContainer(Container, jx_base.Namespace, jx_base.Table):
    """
    A CONTAINER WITH ONLY ONE TABLE
    """
    def __init__(self, name, data, schema=None):
        # TODO: STORE THIS LIKE A CUBE FOR FASTER ACCESS AND TRANSFORMATION
        data = list(unwrap(data))
        Container.__init__(self)
        if schema == None:
            self._schema = get_schema_from_list(name, data)
        else:
            self._schema = schema
        self.name = name
        self.data = data
        self.locker = Lock()  # JUST IN CASE YOU WANT TO DO MORE THAN ONE THING

    @property
    def query_path(self):
        return None

    @property
    def schema(self):
        return self._schema

    @property
    def namespace(self):
        return self

    def last(self):
        """
        :return:  Last element in the list, or Null
        """
        if self.data:
            return self.data[-1]
        else:
            return Null

    def query(self, q):
        q = wrap(q)
        output = self
        if is_aggs(q):
            output = list_aggs(output.data, q)
        else:  # SETOP
            try:
                if q.filter != None or q.esfilter != None:
                    Log.error("use 'where' clause")
            except AttributeError:
                pass

            if q.where is not TRUE and not q.where is TRUE:
                output = output.filter(q.where)

            if q.sort:
                output = output.sort(q.sort)

            if q.select:
                output = output.select(q.select)
        #TODO: ADD EXTRA COLUMN DESCRIPTIONS TO RESULTING SCHEMA
        for param in q.window:
            output.window(param)

        if q.format:
            if q.format == "list":
                return Data(data=output.data, meta={"format": "list"})
            elif q.format == "table":
                head = [c.name for c in output.schema.columns]
                data = [
                    [r if h == "." else r[h] for h in head]
                    for r in output.data
                ]
                return Data(header=head, data=data, meta={"format": "table"})
            elif q.format == "cube":
                head = [c.name for c in output.schema.columns]
                rows = [
                    [r[h] for h in head]
                    for r in output.data
                ]
                data = {h: c for h, c in zip(head, zip(*rows))}
                return Data(
                    data=data,
                    meta={"format": "cube"},
                    edges=[{
                        "name": "rownum",
                        "domain": {"type": "rownum", "min": 0, "max": len(rows), "interval": 1}
                    }]
                )
            else:
                Log.error("unknown format {{format}}", format=q.format)
        else:
            return output

    def update(self, command):
        """
        EXPECTING command == {"set":term, "clear":term, "where":where}
        THE set CLAUSE IS A DICT MAPPING NAMES TO VALUES
        THE where CLAUSE IS A JSON EXPRESSION FILTER
        """
        command = wrap(command)
        command_clear = listwrap(command["clear"])
        command_set = command.set.items()
        command_where = jx.get(command.where)

        for c in self.data:
            if command_where(c):
                for k in command_clear:
                    c[k] = None
                for k, v in command_set:
                    c[k] = v

    def filter(self, where):
        return self.where(where)

    def where(self, where):
        if is_data(where):
            temp = jx_expression_to_function(where)
        elif is_expression(where):
            temp = jx_expression_to_function(where)
        else:
            temp = where

        return ListContainer("from "+self.name, filter(temp, self.data), self.schema)

    def sort(self, sort):
        return ListContainer("sorted "+self.name, jx.sort(self.data, sort, already_normalized=True), self.schema)

    def get(self, select):
        """
        :param select: the variable to extract from list
        :return:  a simple list of the extraction
        """
        if is_list(select):
            return [(d[s] for s in select) for d in self.data]
        else:
            return [d[select] for d in self.data]

    def select(self, select):
        selects = listwrap(select)

        if len(selects) == 1 and is_op(selects[0].value, Variable) and selects[0].value.var == ".":
            new_schema = self.schema
            if selects[0].name == ".":
                return self
        else:
            new_schema = None

        if is_list(select):
            if all(
                is_op(s.value, Variable) and s.name == s.value.var
                for s in select
            ):
                names = set(s.value.var for s in select)
                new_schema = Schema(".", [c for c in self.schema.columns if c.name in names])

            push_and_pull = [(s.name, jx_expression_to_function(s.value)) for s in selects]
            def selector(d):
                output = Data()
                for n, p in push_and_pull:
                    output[n] = unwraplist(p(wrap(d)))
                return unwrap(output)

            new_data = map(selector, self.data)
        else:
            select_value = jx_expression_to_function(select.value)
            new_data = map(select_value, self.data)
            if is_op(select.value, Variable):
                column = copy(first(c for c in self.schema.columns if c.name == select.value.var))
                column.name = '.'
                new_schema = Schema("from " + self.name, [column])

        return ListContainer("from "+self.name, data=new_data, schema=new_schema)

    def window(self, window):
        # _ = window
        jx.window(self.data, window)
        return self

    def having(self, having):
        _ = having
        raise NotImplementedError()

    def format(self, format):
        if format == "table":
            frum = convert.list2table(self.data, self._schema.lookup.keys())
        elif format == "cube":
            frum = convert.list2cube(self.data, self.schema.lookup.keys())
        else:
            frum = self.__data__()

        return frum

    def groupby(self, keys, contiguous=False):
        try:
            keys = listwrap(keys)
            get_key = jx_expression_to_function(keys)
            if not contiguous:
                data = sort_using_key(self.data, key=get_key)

            def _output():
                for g, v in itertools.groupby(data, get_key):
                    group = Data()
                    for k, gg in zip(keys, g):
                        group[k] = gg
                    yield (group, wrap(list(v)))

            return _output()
        except Exception as e:
            Log.error("Problem grouping", e)

    def insert(self, documents):
        self.data.extend(documents)

    def extend(self, documents):
        self.data.extend(documents)

    def __data__(self):
        if first(self.schema.columns).name=='.':
            return wrap({
                "meta": {"format": "list"},
                "data": self.data
            })
        else:
            return wrap({
                "meta": {"format": "list"},
                "data": [{k: unwraplist(v) for k, v in row.items()} for row in self.data]
            })

    def get_columns(self, table_name=None):
        return self.schema.values()

    def add(self, value):
        self.data.append(value)

    def __getitem__(self, item):
        if item < 0 or len(self.data) <= item:
            return Null
        return self.data[item]

    def __iter__(self):
        return (wrap(d) for d in self.data)

    def __len__(self):
        return len(self.data)

    def get_snowflake(self, name):
        if self.name != name:
            Log.error("This container only has table by name of {{name}}", name=name)
        return self

    def get_schema(self, name):
        if self.name != name:
            Log.error("This container only has table by name of {{name}}", name=name)
        return self.schema

    def get_table(self, name):
        if self is name or self.name == name:
            return self
        Log.error("This container only has table by name of {{name}}", name=name)


def _exec(code):
    try:
        temp = None
        exec("temp = " + code)
        return temp
    except Exception as e:
        Log.error("Could not execute {{code|quote}}", code=code, cause=e)


from jx_base.schema import Schema
from jx_python import jx


DUAL = ListContainer(
    name="dual",
    data=[{}],
    schema=Schema(table_name="dual", columns=UniqueIndex(keys=("name",)))
)

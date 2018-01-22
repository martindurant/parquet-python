# encoding: utf-8
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

from copy import copy
from datetime import datetime

from mo_future import text_type
from mo_dots import wrap, Data, FlatList, literal_field
from mo_json.typed_encoder import TYPE_PREFIX
from mo_logs import Log
from pyLibrary import convert
from jx_python import jx
from jx_python.containers import Container
from jx_python.expressions import Variable, Literal
from jx_base.query import QueryOp

INDEX = "__index__"
PARENT = "__parent__"

class DocStore(Container):
    """
    SIMPLE COLUMNAR DATASTORE, EVERYTHING IS INDEXED, WITH QUERY INTERFACE
    HOPE IS IT WILL USE NUMPY
    """
    def __init__(self, uid="_id"):
        self._uid = uid  # COLUMN NAME HOLDING UID
        self._source = []  # ORDERED LIST OF ALL wrapped DOCUMENTS
        self._index = {}  # MAP FROM PROPERTY_NAME -> (VALUE -> set(OF _source INDEXES)
        self._unique_index = {}  # MAP FROM _UID TO _source INDEX

    def add(self, doc):
        doc = wrap(copy(doc))
        _id = doc[self._uid]
        if _id == None:
            _source_index = _id = doc[self._uid] = len(self._source)
        else:
            _source_index = self._unique_index[_id]
            existing = self._source[_source_index]
            self._unindex_values(existing, _source_index)
        self._source.append(doc)
        self._index_values(doc, _source_index)

    def update(self, clear, set, where):
        doc_list = self._filter(where)
        self._update(clear, set, doc_list)

    def upsert(self, clear, set, where):
        doc_list = self._filter(where)
        if not doc_list:
            self.add(set)
        else:
            self._update(clear, set, doc_list)

    def _update(self, clear, set, doc_list):
        for _source_index in doc_list:
            existing = self._source[_source_index]
            self._unindex_values(existing, _source_index)
            for c in clear:
                existing[c] = None
            for k, v in set.items():
                existing[k] = v
            self._index_values(existing, _source_index)

    def _index_values(self, doc, start_index, parent_index=-1, prefix=""):
        curr_index = doc[INDEX] = start_index
        doc[PARENT] = parent_index
        _index = self._index

        for k, v in doc.items():
            k = literal_field(k)
            _type = _type_map[v.__class__]
            if _type == "object":
                self._index_values(v, start_index, prefix=k + ".")
                v = "."
            elif _type == "nested":
                for vv in v:
                    curr_index = self._index_values(vv, curr_index + 1, start_index, prefix=k + ".")
                _type = "object"
                v = "."

            typed_key = k + "." + TYPE_PREFIX + _type
            i = _index.get(typed_key)
            if i is None:
                i = _index[typed_key] = {}
            j = i.get(v)
            if j is None:
                j = i[v] = set()
            j |= {start_index}
        return curr_index

    def _unindex_values(self, existing, _source_index):
        self._unique_index[existing[self._uid]] = None
        for k, v in existing.leaves():
            self._index[k][v] -= {_source_index}

    def query(self, query):
        query = QueryOp.wrap(query)
        short_list = self._filter(query.where)
        if query.sort:
            short_list = self._sort(query.sort)

        if isinstance(query.select, list):
            accessors = map(jx.get, query.select.value)

        if query.window:
            for w in query.window:
                window_list = self._filter(w.where)

    def _edges(self, short_list, edges):
        edge_values = self._index_columns(edges)

    def _index_columns(self, columns):
        # INDEX ALL COLUMNS, ESPECIALLY THOSE FUNCTION RESULTS
        indexed_values = [None]*len(columns)
        for i, s in enumerate(columns):
            index = self._index.get(s.value, None)
            if index is not None:
                indexed_values[i]=index
                continue

            function_name = value2json(s.value.__data__(), sort_keys=True)
            index = self._index.get(function_name, None)
            indexed_values[i]=index
            if index is not None:
                continue

            indexed_values[i] = index = self._index[function_name] = {}
            accessor = jx.get(s.value)
            for k, ii in self._unique_index.items():
                v = accessor(self._source[ii])
                j = index.get(v)
                if j is None:
                    j = index[v] = set()
                j |= {ii}
        return indexed_values

    def _sort(self, short_list, sorts):
        """
        TAKE SHORTLIST, RETURN IT SORTED
        :param short_list:
        :param sorts: LIST OF SORTS TO PERFORM
        :return:
        """

        sort_values = self._index_columns(sorts)

        # RECURSIVE SORTING
        output = []
        def _sort_more(short_list, i, sorts):
            if len(sorts) == 0:
                output.extend(short_list)

            sort = sorts[0]

            index = self._index[sort_values[i]]
            if sort.sort == 1:
                sorted_keys = sorted(index.keys())
            elif sort.sort == -1:
                sorted_keys = reversed(sorted(index.keys()))
            else:
                sorted_keys = list(index.keys())

            for k in sorted_keys:
                self._sort(index[k] & short_list, i + 1, sorts[1:])

        _sort_more(short_list, 0, sorts)
        return output

    def filter(self, where):
        return self.where(where)

    def where(self, where):
        return self.query({"from": self, "where": where})

    def sort(self, sort):
        return self.query({"from": self, "sort": sort})

    def select(self, select):
        return self.query({"from": self, "select": select})

    def window(self, window):
        return self.query({"from": self, "window": window})

    def having(self, having):
        _ = having
        Log.error("not implemented")

    def format(self, format):
        if format == "list":
            return {
                "meta": {"format": "list"},
                "data": [self._source[i] for i in self._unique_index.values()]
            }
        elif format == "table":
            columns = list(self._index.keys())
            data = [[self._source[i].get(c, None) for c in columns] for i in self._unique_index.values()]
            return {
                "meta": {"format": "table"},
                "header": columns,
                "data": data
            }
        elif format == "cube":
            Log.error("not supported")

    def get_leaves(self, table_name):
        return {"name":c for c in self._index.keys()}

    def _filter(self, where):
        return filters[where.name](self, where)

    def _eq(self, op):
        if isinstance(op.lhs, Variable) and isinstance(op.rhs, Literal):
            return copy(self._index[op.lhs][op.rhs])

    def _and(self, op):
        if not op.terms:
            return self._unique_index.values()

        agg = filters[op.name](self, op.terms[0])
        for t in op.terms[1:]:
            agg &= filters[op.name](self, t)
        return agg

    def _true(self, op):
        return self._unique_index.values()


filters={
    "eq": DocStore._eq,
    "and": DocStore._and,
    "true": DocStore._true
}

_type_map = {
    text_type: "text",
    int: "long",
    float: "real",
    datetime: "real",
    list: "nested",
    FlatList: "nested",
    dict: "object",
    Data: "object"
}

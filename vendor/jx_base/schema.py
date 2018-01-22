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

from copy import copy

from jx_base import STRUCT, NESTED, PRIMITIVE, OBJECT, EXISTS
from mo_dots import join_field, split_field, Null, startswith_field, set_default, wrap
from mo_json.typed_encoder import unnest_path, untype_path, NESTED_TYPE
from mo_logs import Log


def _indexer(columns, query_path):
    all_names = set(unnest_path(n) for c in columns for n in c.names.values()) | {"."}

    lookup_leaves = {}
    for full_name in all_names:
        for c in columns:
            cname = c.names[query_path]
            nfp = unnest_path(cname)
            if (
                startswith_field(nfp, full_name) and
                c.type not in [EXISTS, OBJECT] and
                (c.es_column != "_id" or full_name == "_id") and
                startswith_field(nfp, full_name)
            ):
                cs = lookup_leaves.setdefault(full_name, set())
                cs.add(c)
                cs = lookup_leaves.setdefault(untype_path(full_name), set())
                cs.add(c)

    relative_lookup = {}
    for c in columns:
        try:
            cname = c.names[query_path]
            cs = relative_lookup.setdefault(cname, set())
            cs.add(c)

            ucname = untype_path(cname)
            cs = relative_lookup.setdefault(ucname, set())
            cs.add(c)
        except Exception as e:
            Log.error("Should not happen", cause=e)

    if query_path != ".":
        absolute_lookup, more_leaves = _indexer(columns, ".")
        for k, cs in absolute_lookup.items():
            if k not in relative_lookup:
                relative_lookup[k] = cs
        for k, cs in more_leaves.items():
            if k not in lookup_leaves:
                lookup_leaves[k] = cs

    return relative_lookup, lookup_leaves


class Schema(object):
    """
    A Schema MAPS ALL COLUMNS IN SNOWFLAKE FROM NAME TO COLUMN INSTANCE
    """

    def __init__(self, table_name, columns):
        """
        :param table_name: THE FACT TABLE
        :param query_path: PATH TO ARM OF SNOWFLAKE
        :param columns: ALL COLUMNS IN SNOWFLAKE
        """
        self._columns = copy(columns)
        table_path = split_field(table_name)
        self.table = table_path[0]  # USED AS AN EXPLICIT STATEMENT OF PERSPECTIVE IN THE DATABASE
        query_path = join_field(table_path[1:])  # TODO: REPLACE WITH THE nested_path ARRAY
        if query_path == ".":
            self.query_path = query_path
        else:
            query_path += "."+NESTED_TYPE
            self.query_path = [c for c in columns if c.type == NESTED and c.names["."] == query_path][0].es_column
        self.lookup, self.lookup_leaves = _indexer(columns, self.query_path)

    def __getitem__(self, column_name):
        cs = self.lookup.get(column_name)
        if cs:
            return list(cs)
        else:
            return [wrap({"es_column": column_name})]

    def items(self):
        return self.lookup.items()

    def get_column(self, name, table=None):
        return self.lookup[name]

    def get_column_name(self, column):
        """
        RETURN THE COLUMN NAME, FROM THE PERSPECTIVE OF THIS SCHEMA
        :param column:
        :return: NAME OF column
        """
        return column.names[self.query_path]

    def values(self, name):
        """
        RETURN VALUES FOR THE GIVEN PATH NAME
        :param name:
        :return:
        """
        full_name = unnest_path(name)
        return list(set([
            c
            for c in self.lookup.get(full_name, Null)
            if c.type in PRIMITIVE and (c.es_column != "_id")  # MULTIVALUES ARE LEGIT, SO NESTED IS FINE: and self.query_path == c.nested_path[0]
        ]))

    def leaves(self, name, meta=False):
        """
        RETURN LEAVES OF GIVEN PATH NAME
        pull leaves, considering query_path and namespace
        pull all first-level properties
        pull leaves, including parent leaves
        pull the head of any tree by name
        :param name:
        :return:
        """

        return list(self.lookup_leaves.get(unnest_path(name), Null))

    def map_to_es(self):
        """
        RETURN A MAP FROM THE NAME SPACE TO THE es_column NAME
        """
        full_name = self.query_path
        return set_default(
            {
                c.names[full_name]: c.es_column
                for k, cs in self.lookup.items()
                # if startswith_field(k, full_name)
                for c in cs if c.type not in STRUCT
            },
            {
                c.names["."]: c.es_column
                for k, cs in self.lookup.items()
                # if startswith_field(k, full_name)
                for c in cs if c.type not in STRUCT
            }
        )

    @property
    def columns(self):
        return copy(self._columns)



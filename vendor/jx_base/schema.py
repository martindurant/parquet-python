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

from mo_future import is_text, is_binary
from copy import copy

from mo_dots import Null, relative_field, set_default, startswith_field, wrap
from mo_json import EXISTS, NESTED, OBJECT, STRUCT
from mo_json.typed_encoder import unnest_path, untype_path
from mo_logs import Log


class Schema(object):
    """
    A Schema MAPS COLUMN NAMES OF A SINGLE TABLE TO COLUMN INSTANCES THAT MATCH
    """

    def __init__(self, table_name, columns):
        """
        :param table_name: A FULL NAME FOR THIS TABLE (NOT USED)
        :param columns: ALL COLUMNS IN SNOWFLAKE
        """
        self._columns = copy(columns)
        self.table = table_name
        self.query_path = "."
        self.lookup, self.lookup_leaves, self.lookup_variables = _indexer(columns, self.query_path)

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

    @property
    def columns(self):
        return self._columns

    def get_column_name(self, column):
        """
        RETURN THE COLUMN NAME, FROM THE PERSPECTIVE OF THIS SCHEMA
        :param column:
        :return: NAME OF column
        """
        return relative_field(column.name, query_path)

    def values(self, name):
        """
        RETURN VALUES FOR THE GIVEN PATH NAME
        :param name:
        :return:
        """
        return list(self.lookup_variables.get(unnest_path(name), Null))

    def leaves(self, name):
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
        RETURN A MAP FROM THE NAMESPACE TO THE es_column NAME
        """
        full_name = self.query_path
        return set_default(
            {
                relative_field(c.name, full_name): c.es_column
                for k, cs in self.lookup.items()
                # if startswith_field(k, full_name)
                for c in cs if c.jx_type not in STRUCT
            },
            {
                c.name: c.es_column
                for k, cs in self.lookup.items()
                # if startswith_field(k, full_name)
                for c in cs if c.jx_type not in STRUCT
            }
        )

    @property
    def columns(self):
        return copy(self._columns)


def _indexer(columns, query_path):
    all_names = set(unnest_path(c.name) for c in columns) | {"."}

    lookup_leaves = {}  # ALL LEAF VARIABLES
    for full_name in all_names:
        for c in columns:
            cname = relative_field(c.name, query_path)
            nfp = unnest_path(cname)
            if (
                startswith_field(nfp, full_name) and
                c.es_type not in [EXISTS, OBJECT, NESTED] and
                (c.es_column != "_id" or full_name == "_id")
            ):
                cs = lookup_leaves.setdefault(full_name, set())
                cs.add(c)
                cs = lookup_leaves.setdefault(untype_path(full_name), set())
                cs.add(c)

    lookup_variables = {}  # ALL NOT-NESTED VARIABLES
    for full_name in all_names:
        for c in columns:
            cname = relative_field(c.name, query_path)
            nfp = unnest_path(cname)
            if (
                startswith_field(nfp, full_name) and
                c.es_type not in [EXISTS, OBJECT] and
                (c.es_column != "_id" or full_name == "_id") and
                startswith_field(c.nested_path[0], query_path)
            ):
                cs = lookup_variables.setdefault(full_name, set())
                cs.add(c)
                cs = lookup_variables.setdefault(untype_path(full_name), set())
                cs.add(c)

    relative_lookup = {}
    for c in columns:
        try:
            cname = relative_field(c.name, query_path)
            cs = relative_lookup.setdefault(cname, set())
            cs.add(c)

            ucname = untype_path(cname)
            cs = relative_lookup.setdefault(ucname, set())
            cs.add(c)
        except Exception as e:
            Log.error("Should not happen", cause=e)

    if query_path != ".":
        # ADD ABSOLUTE NAMES TO THE NAMESAPCE
        absolute_lookup, more_leaves, more_variables = _indexer(columns, ".")
        for k, cs in absolute_lookup.items():
            if k not in relative_lookup:
                relative_lookup[k] = cs
        for k, cs in more_leaves.items():
            if k not in lookup_leaves:
                lookup_leaves[k] = cs
        for k, cs in more_variables.items():
            if k not in lookup_variables:
                lookup_variables[k] = cs

    return relative_lookup, lookup_leaves, lookup_variables


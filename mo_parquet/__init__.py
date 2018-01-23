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

from jx_base import OBJECT, NESTED, STRING
from mo_dots import concat_field
from mo_json.typed_encoder import NESTED_TYPE
from mo_parquet.schema import SchemaTree, get_length, get_repetition_type, merge_schema_element, python_type_to_all_types
from mo_parquet.table import Table

REQUIRED = 'required'
OPTIONAL = 'optional'
REPEATED = 'repeated'


def rows_to_columns(data, schema):
    """
    :param data: array of objects
    :param schema: Known schema, will be extended to include all properties found in data
    :return: Table
    """

    new_schema = []

    all_leaves = schema.leaves
    values = {full_name: [] for full_name in all_leaves}
    rep_levels = {full_name: [] for full_name in all_leaves}
    def_levels = {full_name: [] for full_name in all_leaves}

    def _none_to_column(schema, path, rep_level, counters):
        if schema:
            for name, sub_schema in schema.items():
                new_path = concat_field(path, name)
                _none_to_column(sub_schema, new_path, rep_level, counters)
        else:
            values[path].append(None)
            rep_levels[path].append(rep_level)
            def_levels[path].append(len(counters) - 1)

    def _value_to_column(value, schema, path, counters):
        ptype = type(value)
        dtype, jtype, itype = python_type_to_all_types[ptype]
        if jtype is NESTED:
            new_path = concat_field(path, NESTED_TYPE)
            sub_schema = schema.more.get(NESTED_TYPE)
            if not sub_schema:
                sub_schema = schema.more[NESTED_TYPE] = SchemaTree()

            if not value:
                _none_to_column(sub_schema, new_path, get_rep_level(counters), counters)
            else:
                for k, new_value in enumerate(value):
                    new_counters = counters + (k,)
                    _value_to_column(new_value, sub_schema, new_path, new_counters)
        elif jtype is OBJECT:
            if not value:
                _none_to_column(schema, path, get_rep_level(counters), counters)
            else:
                for name, sub_schema in schema.more.items():
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    _value_to_column(new_value, sub_schema, new_path, counters)

                for name in set(value.keys()) - set(schema.more.keys()):
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    sub_schema = schema.more[name] = SchemaTree()
                    _value_to_column(new_value, sub_schema, new_path, counters)
        else:
            typed_name = concat_field(path, itype)
            if jtype is STRING:
                value = value.encode('utf8')
            element, is_new = merge_schema_element(schema.values.get(itype), typed_name, value, ptype, dtype, jtype, itype)
            if is_new:
                schema.values[itype] = element
                new_schema.append(element)
                values[typed_name] = [None] * counters[0]
                rep_levels[typed_name] = [0] * counters[0]
                def_levels[typed_name] = [0] * counters[0]
            values[typed_name].append(value)
            rep_levels[typed_name].append(get_rep_level(counters))
            def_levels[typed_name].append(len(counters) - 1)

    for rownum, new_value in enumerate(data):
        _value_to_column(new_value, schema, '.', (rownum,))

    return Table(values, rep_levels, def_levels, len(data), schema), new_schema


def get_rep_level(counters):
    for rep_level, c in reversed(list(enumerate(counters))):
        if c > 0:
            return rep_level
    return 0  # SHOULD BE -1 FOR MISSING RECORD, BUT WE WILL ASSUME THE RECORD EXISTS


def assemble(values, rep_levels, def_levels, schema):
    max = schema.max_definition_level()+1

    def _add(value, rep_level, def_level, parents):
        if def_level == len(parents):
            new_parents = parents[0:rep_level + 1]
            for _ in range(rep_level, max):
                new_child = []
                new_parents[-1].append(new_child)
                new_parents.append(new_child)
            new_parents[-1].append(value)
        else:
            new_parents = parents[0:def_level + 1]
            new_parents.append(None)

    rows = []
    parents = [rows]
    for value, rep_level, def_level in zip(values, def_levels, rep_levels):
        _add(parents, value, rep_level, def_level)



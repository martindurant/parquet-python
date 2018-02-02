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

from jx_base import OBJECT, NESTED, STRING, PRIMITIVE, python_type_to_json_type
from mo_dots import concat_field
from mo_logs import Log
from mo_parquet.schema import SchemaTree, get_length, get_repetition_type, merge_schema_element, python_type_to_all_types, OPTIONAL, REQUIRED, REPEATED
from mo_parquet.table import Table


def rows_to_columns(data, schema=None):
    """
    :param data: array of objects
    :param schema: Known schema, will be extended to include all properties found in data
    :return: Table
    """
    if not schema:
        schema = SchemaTree()
    new_schema = []

    all_leaves = schema.leaves
    values = {full_name: [] for full_name in all_leaves}
    reps = {full_name: [] for full_name in all_leaves}
    defs = {full_name: [] for full_name in all_leaves}

    def _none_to_column(schema, path, rep_level, def_level):
        for full_path in schema.leaves:
            reps[full_path].append(rep_level)
            defs[full_path].append(def_level)

    def _value_to_column(value, schema, path, counters, def_level):
        ptype = type(value)
        ntype, dtype, ltype, jtype, itype, byte_width = python_type_to_all_types[ptype]

        if jtype is NESTED:
            if schema.element.repetition_type != REPEATED:
                Log.error("Expecting {{path|quote}} to be repeated", path=path)

            new_path = path
            if not value:
                _none_to_column(schema, new_path, get_rep_level(counters), def_level)
            else:
                sub_schema = schema.more.get('.')
                if not sub_schema:
                    sub_schema = SchemaTree()  # ALL VALUES IN REPEATED MUST EXIST
                    sub_schema.more = schema.more

                for k, new_value in enumerate(value):
                    new_counters = counters + (k,)
                    _value_to_column(new_value, sub_schema, new_path, new_counters, def_level+1)
        elif jtype is OBJECT:
            if value is None:
                if schema.element.repetition_type == REQUIRED:
                    Log.error("{{path|quote}} is required", path=path)
                _none_to_column(schema, path, get_rep_level(counters), def_level)
            else:
                if schema.element.repetition_type == REPEATED:
                    Log.error("Expecting {{path|quote}} to be repeated", path=path)

                if schema.element.repetition_type == REQUIRED:
                    new_def_level = def_level
                else:
                    new_def_level = def_level+1

                for name, sub_schema in schema.more.items():
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    _value_to_column(new_value, sub_schema, new_path, counters, new_def_level)

                for name in set(value.keys()) - set(schema.more.keys()):
                    if schema.locked:
                        Log.error("{{path}} is not allowed in the schema", path=path)
                    new_path = concat_field(path, name)
                    new_value = value.get(name, None)
                    sub_schema = schema.add(
                        new_path,
                        REPEATED if isinstance(new_value, list) else OPTIONAL,
                        type(new_value)
                    )
                    if python_type_to_json_type[type(new_value)] in PRIMITIVE:
                        values[new_path] = []
                        reps[new_path] = [0] * counters[0]
                        defs[new_path] = [0] * counters[0]
                    _value_to_column(new_value, sub_schema, new_path, counters, new_def_level)
        else:
            if jtype is STRING:
                value = value.encode('utf8')
            merge_schema_element(schema.element, path, value, ptype, ltype, dtype, jtype, itype, byte_width)
            values[path].append(value)
            if schema.element.repetition_type == REQUIRED:
                reps[path].append(get_rep_level(counters))
                defs[path].append(def_level)
            else:
                reps[path].append(get_rep_level(counters))
                defs[path].append(def_level+1)

    for rownum, new_value in enumerate(data):
        try:
            _value_to_column(new_value, schema, '.', (rownum,), 0)
        except Exception as e:
            Log.error("can not encode {{row|json}}", row=new_value, cause=e)

    return Table(values, reps, defs, len(data), schema)


def get_rep_level(counters):
    for rep_level, c in reversed(list(enumerate(counters))):
        if c > 0:
            return rep_level
    return 0  # SHOULD BE -1 FOR MISSING RECORD, BUT WE WILL ASSUME THE RECORD EXISTS


def assemble(values, reps, defs, schema):
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
    for value, rep_level, def_level in zip(values, defs, reps):
        _add(parents, value, rep_level, def_level)



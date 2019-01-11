# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

import numpy as np

from mo_dots import concat_field, startswith_field
from mo_json import NESTED, OBJECT, PRIMITIVE, STRING, python_type_to_json_type
from mo_logs import Log
from mo_parquet.schema import OPTIONAL, REPEATED, REQUIRED, SchemaTree, get_length, get_repetition_type, merge_schema_element, python_type_to_all_types
from mo_parquet.table import Table


def rows_to_columns(data, schema=None):
    """
    :param data: array of objects
    :param schema: Known schema, will be extended to include all properties found in data
    :return: Table
    """
    if not schema:
        schema = SchemaTree()

    all_leaves = schema.leaves
    values = {full_name: [] for full_name in all_leaves}
    reps = {full_name: [] for full_name in all_leaves}
    defs = {full_name: [] for full_name in all_leaves}

    def _none_to_column(schema, path, rep_level, def_level):
        for full_path in all_leaves:
            if startswith_field(full_path, path):
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
                try:
                    new_schema = schema.more.get('.')
                    if not new_schema:
                        # DEFAULT TO REQUIRED ENTRIES
                        new_schema = schema
                        schema.element.repetition_type = REQUIRED
                    for k, new_value in enumerate(value):
                        new_counters = counters + (k,)
                        _value_to_column(new_value, new_schema, new_path, new_counters, def_level+1)
                finally:
                    schema.element.repetition_type = REPEATED
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


class DremelParser(object):
    """
    WE WILL 'PARSE' THE DREMEL ENCODED ARRAYS INTO DATA
    """

    def __init__(self, table, rep_depth=0, columns=None):
        # TODO: USE THE columns TO BUILD A RESTRICTED schema
        # self.schema = table.schema.select(columns)
        self.schema = schema = table.schema

        self.rep_depth = rep_depth
        self.values = table.values
        # DESTRUCTIVELY USE defs TO COUNT-DOWN TO ZERO
        self.defs = {c: table.schema.max_definition_level(c)-table.defs[c] for c in schema.columns}
        # WE ARE DESTRUCTIVE TO THE REP COUNTERS
        self.reps = {c: np.copy(table.reps[c]) for c in schema.columns}
        self.vi = {c: 0 for c in schema.columns}
        self.i = {c: 0 for c in schema.columns}

    def next_row(self):
        return self.assemble_required(self.schema)

    def assemble_required(self, schema):
        if self.i.get(schema.name) is None or self.defs[schema.name][self.i[schema.name]]:
            output = {}
            for name, sub_schema in schema.more.items():
                output[name] = assemble[sub_schema.repetition_type](self, sub_schema)
            return output
        else:
            output = self.values[schema.name]
            self.vi[schema.name] += 1
            self.i[schema.name] += 1
            return output

    def assemble_optional(self, schema):
        if self.i.get(schema.name) is None or self.defs[schema.name][self.i[schema.name]]:
            output = {}
            for name, sub_schema in schema.more.items():
                self.defs[schema.name][self.i[schema.name]] -= 1
                output[name] = assemble[sub_schema.repetition_type](self, sub_schema)
            return output
        else:
            output = self.values[schema.name]
            self.vi[schema.name] += 1
            self.i[schema.name] += 1
            return output

    def assemble_repeated(self, schema):
        if self.i.get(schema.name) is None or self.defs[schema.name][self.i[schema.name]]:
            output = []
            columns = schema.columns



            is_done = {c: False for c in columns}
            while not all(is_done[c] for c in columns):

                row = {}
                for name, sub_schema in schema.more.items():
                    value = self.assemble_required(sub_schema)
                    row[name] = value
                output.append(row)
            return output
        else:
            output = self.values[schema.name][self.i[schema.name]]
            self.vi[schema.name] += 1
            self.i[schema.name] += 1
            return output


assemble = {
    REQUIRED: DremelParser.assemble_required,
    OPTIONAL: DremelParser.assemble_optional,
    REPEATED: DremelParser.assemble_repeated
}



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

import pandas as pd

from mo_dots import split_field, startswith_field, coalesce
from mo_future import text_type


class Table(object):
    """
    REPRESENT A DATA CUBE
    """

    def __init__(self, values, rep_levels, def_levels, num_rows, schema, max_definition_level=None):
        """
        :param values: dict from full name to list of values
        :param rep_levels:  dict from full name to list of values
        :param def_levels: dict from full name to list of values
        :param num_rows: number of rows in the dataset
        :param schema: The complete SchemaTree
        """
        self.values = pd.DataFrame.from_dict(values)
        self.rep_levels = pd.DataFrame.from_dict(rep_levels)
        self.def_levels = pd.DataFrame.from_dict(def_levels)
        self.num_rows = num_rows
        self.schema = schema
        self.max_definition_level = max_definition_level or schema.max_definition_level()

    def __getattr__(self, item):
        return getattr(self.values, item)

    def get_column(self, item):
        sub_schema = self.schema
        for n in split_field(item):
            if n in sub_schema.more:
                sub_schema = sub_schema.more.get(n)
            else:
                sub_schema = sub_schema.values.get(n)

        return Column(
            item,
            self.values[item],
            self.rep_levels[item],
            self.def_levels[item],
            self.num_rows,
            sub_schema,
            self.max_definition_level
        )

    @property
    def columns(self):
        return self.values.keys()

    def __getitem__(self, item):
        if isinstance(item, text_type):
            sub_schema = self.schema
            for n in split_field(item):
                if n in sub_schema.more:
                    sub_schema = sub_schema.more.get(n)
                else:
                    sub_schema = sub_schema.values.get(n)

            return Table(
                {k: v for k, v in self.values.items() if startswith_field(k, item)},
                {k: v for k, v in self.rep_levels.items() if startswith_field(k, item)},
                {k: v for k, v in self.def_levels.items() if startswith_field(k, item)},
                self.num_rows,
                sub_schema,
                self.max_definition_level
            )
        elif isinstance(item, slice):
            start = coalesce(item.start, 0)
            stop = coalesce(item.stop, self.num_rows)

            if start == 0 and stop == self.num_rows:
                return self

            first = 0
            last = 0
            counter = 0
            for i, r in enumerate(self.rep_levels):
                if counter == start:
                    first = i
                elif counter == stop:
                    last = i
                    break
                if r == 0:
                    counter += 1

            return Table(
                {k: v[first:last] for k, v in self.values.items()},
                {k: v[first:last] for k, v in self.rep_levels.items()},
                {k: v[first:last] for k, v in self.def_levels.items()},
                stop - start,
                self.schema
            )

    def __len__(self):
        return self.num_rows


class Column(object):
    """
    REPRESENT A DATA FRAME
    """

    def __init__(self, name, values, rep_levels, def_levels, num_rows, schema, max_definition_level):
        """
        :param values: MAP FROM NAME TO LIST OF PARQUET VALUES
        :param schema:
        """
        self.name = name
        self.values = values
        self.rep_levels = rep_levels
        self.def_levels = def_levels
        self.num_rows = num_rows
        self.schema = schema
        self.max_definition_level = max_definition_level

    def __len__(self):
        return self.num_rows


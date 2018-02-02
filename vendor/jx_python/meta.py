# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from copy import copy
from datetime import date
from datetime import datetime

from mo_future import text_type, long, PY2
from jx_base import STRUCT

from jx_python import jx
from mo_collections import UniqueIndex
from mo_dots import Data, concat_field, get_attr, listwrap, unwraplist, NullType, FlatList
from mo_dots import split_field, join_field, ROOT_PATH
from mo_dots import wrap
from mo_json.typed_encoder import untype_path, unnest_path
from mo_logs import Log
from mo_threads import Lock
from mo_future import none_type

from jx_base.container import Container
from jx_base.schema import Schema
from mo_times.dates import Date
from pyLibrary.meta import DataClass

singlton = None




def metadata_columns():
    return wrap(
        [
            Column(
                names={".":c},
                es_index="meta.columns",
                es_column=c,
                type="string",
                nested_path=ROOT_PATH
            )
            for c in [
                "type",
                "nested_path",
                "es_column",
                "es_index"
            ]
        ] + [
            Column(
                es_index="meta.columns",
                names={".":c},
                es_column=c,
                type="object",
                nested_path=ROOT_PATH
            )
            for c in [
                "names",
                "domain",
                "partitions"
            ]
        ] + [
            Column(
                names={".": c},
                es_index="meta.columns",
                es_column=c,
                type="long",
                nested_path=ROOT_PATH
            )
            for c in [
                "count",
                "cardinality"
            ]
        ] + [
            Column(
                names={".": "last_updated"},
                es_index="meta.columns",
                es_column="last_updated",
                type="time",
                nested_path=ROOT_PATH
            )
        ]
    )


def metadata_tables():
    return wrap(
        [
            Column(
                names={".": c},
                es_index="meta.tables",
                es_column=c,
                type="string",
                nested_path=ROOT_PATH
            )
            for c in [
                "name",
                "url",
                "query_path"
            ]
        ]+[
            Column(
                names={".": "timestamp"},
                es_index="meta.tables",
                es_column="timestamp",
                type="integer",
                nested_path=ROOT_PATH
            )
        ]
    )


class Table(DataClass("Table", [
    "name",
    "url",
    "query_path",
    "timestamp"
])):
    @property
    def columns(self):
        return singlton.get_columns(table_name=self.name)


Column = DataClass(
    "Column",
    [
        # "table",
        "names",  # MAP FROM TABLE NAME TO COLUMN NAME (ONE COLUMN CAN HAVE MULTIPLE NAMES)
        "es_column",
        "es_index",
        # "es_type",
        "type",
        {"name": "useSource", "default": False},
        {"name": "nested_path", "nulls": True},  # AN ARRAY OF PATHS (FROM DEEPEST TO SHALLOWEST) INDICATING THE JSON SUB-ARRAYS
        {"name": "count", "nulls": True},
        {"name": "cardinality", "nulls": True},
        {"name": "multi", "nulls": True},
        {"name": "partitions", "nulls": True},
        {"name": "last_updated", "nulls": True}
    ]
)


class ColumnList(Container):
    """
    OPTIMIZED FOR THE PARTICULAR ACCESS PATTERNS USED
    """

    def __init__(self):
        self.data = {}  # MAP FROM ES_INDEX TO (abs_column_name to COLUMNS)
        self.locker = Lock()
        self.count = 0
        self.meta_schema = None

    def find(self, es_index, abs_column_name):
        if "." in es_index and not es_index.startswith("meta."):
            Log.error("unlikely index name")
        if not abs_column_name:
            return [c for cs in self.data.get(es_index, {}).values() for c in cs]
        else:
            return self.data.get(es_index, {}).get(abs_column_name, [])

    def insert(self, columns):
        for column in columns:
            self.add(column)

    def add(self, column):
        columns_for_table = self.data.setdefault(column.es_index, {})
        abs_cname = column.names["."]
        _columns = columns_for_table.get(abs_cname)
        if not _columns:
            _columns = columns_for_table[abs_cname] = []
        _columns.append(column)
        self.count += 1

    def __iter__(self):
        for t, cs in self.data.items():
            for c, css in cs.items():
                for column in css:
                    yield column

    def __len__(self):
        return self.count

    def update(self, command):
        try:
            command = wrap(command)
            eq = command.where.eq
            if eq.es_index:
                columns = self.find(eq.es_index, eq.name)
                columns = [c for c in columns if all(get_attr(c, k) == v for k, v in eq.items())]
            else:
                columns = list(self)
                columns = jx.filter(columns, command.where)

            for col in list(columns):
                for k in command["clear"]:
                    if k == ".":
                        columns.remove(col)
                    else:
                        col[k] = None

                for k, v in command.set.items():
                    col[k] = v
        except Exception as e:
            Log.error("should not happen", cause=e)

    def query(self, query):
        query.frum = self.__iter__()
        output = jx.run(query)

        return output

    def groupby(self, keys):
        return jx.groupby(self.__iter__(), keys)

    @property
    def schema(self):
        return wrap({k: set(v) for k, v in self.data["meta.columns"].items()})

    def denormalized(self):
        """
        THE INTERNAL STRUCTURE FOR THE COLUMN METADATA IS VERY DIFFERENT FROM
        THE DENORMALIZED PERSPECITVE. THIS PROVIDES THAT PERSPECTIVE FOR QUERIES
        """
        output = [
            {
                "table": concat_field(c.es_index, untype_path(table)),
                "name": untype_path(name),
                "cardinality": c.cardinality,
                "es_column": c.es_column,
                "es_index": c.es_index,
                "last_updated": c.last_updated,
                "count": c.count,
                "nested_path": [unnest_path(n) for n in c.nested_path],
                "type": c.type
            }
            for tname, css in self.data.items()
            for cname, cs in css.items()
            for c in cs
            if c.type not in STRUCT # and c.es_column != "_id"
            for table, name in c.names.items()
        ]
        if not self.meta_schema:
            self.meta_schema = get_schema_from_list("meta\\.columns", output)

        from jx_python.containers.list_usingPythonList import ListContainer
        return ListContainer("meta\\.columns", data=output, schema=self.meta_schema)


def get_schema_from_list(table_name, frum):
    """
    SCAN THE LIST FOR COLUMN TYPES
    """
    columns = UniqueIndex(keys=("names.\\.",))
    _get_schema_from_list(frum, ".", prefix_path=[], nested_path=ROOT_PATH, columns=columns)
    return Schema(table_name=table_name, columns=columns)


def _get_schema_from_list(frum, table_name, prefix_path, nested_path, columns):
    """
    :param frum: The list
    :param table_name: Name of the table this list holds records for
    :param prefix_path: parent path
    :param nested_path: each nested array, in reverse order
    :param columns: map from full name to column definition
    :return:
    """

    for d in frum:
        row_type = _type_to_name[d.__class__]
        if row_type != "object":
            full_name = join_field(prefix_path)
            column = columns[full_name]
            if not column:
                column = Column(
                    names={table_name: full_name},
                    es_column=full_name,
                    es_index=".",
                    type="undefined",
                    nested_path=nested_path
                )
                columns.add(column)
            column.type = _merge_type[column.type][row_type]
        else:
            for name, value in d.items():
                full_name = join_field(prefix_path + [name])
                column = columns[full_name]
                if not column:
                    column = Column(
                        names={table_name: full_name},
                        es_column=full_name,
                        es_index=".",
                        type="undefined",
                        nested_path=nested_path
                    )
                    columns.add(column)
                if isinstance(value, list):
                    if len(value) == 0:
                        this_type = "undefined"
                    elif len(value) == 1:
                        this_type = _type_to_name[value[0].__class__]
                    else:
                        this_type = _type_to_name[value[0].__class__]
                        if this_type == "object":
                            this_type = "nested"
                else:
                    this_type = _type_to_name[value.__class__]
                new_type = _merge_type[column.type][this_type]
                column.type = new_type

                if this_type == "object":
                    _get_schema_from_list([value], table_name, prefix_path + [name], nested_path, columns)
                elif this_type == "nested":
                    np = listwrap(nested_path)
                    newpath = unwraplist([join_field(split_field(np[0])+[name])]+np)
                    _get_schema_from_list(value, table_name, prefix_path + [name], newpath, columns)


_type_to_name = {
    none_type: "undefined",
    NullType: "undefined",
    bool: "boolean",
    str: "string",
    text_type: "string",
    int: "integer",
    float: "double",
    Data: "object",
    dict: "object",
    set: "nested",
    list: "nested",
    FlatList: "nested",
    Date: "double",
    datetime: "double",
    date: "double"
}

if PY2:
    _type_to_name[long] = "integer"

_merge_type = {
    "undefined": {
        "undefined": "undefined",
        "boolean": "boolean",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": "object",
        "nested": "nested"
    },
    "boolean": {
        "undefined": "boolean",
        "boolean": "boolean",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "integer": {
        "undefined": "integer",
        "boolean": "integer",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "long": {
        "undefined": "long",
        "boolean": "long",
        "integer": "long",
        "long": "long",
        "float": "double",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "float": {
        "undefined": "float",
        "boolean": "float",
        "integer": "float",
        "long": "double",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "double": {
        "undefined": "double",
        "boolean": "double",
        "integer": "double",
        "long": "double",
        "float": "double",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "string": {
        "undefined": "string",
        "boolean": "string",
        "integer": "string",
        "long": "string",
        "float": "string",
        "double": "string",
        "string": "string",
        "object": None,
        "nested": None
    },
    "object": {
        "undefined": "object",
        "boolean": None,
        "integer": None,
        "long": None,
        "float": None,
        "double": None,
        "string": None,
        "object": "object",
        "nested": "nested"
    },
    "nested": {
        "undefined": "nested",
        "boolean": None,
        "integer": None,
        "long": None,
        "float": None,
        "double": None,
        "string": None,
        "object": "nested",
        "nested": "nested"
    }
}


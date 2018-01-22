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

from fastparquet.parquet_thrift.parquet.ttypes import Type, FieldRepetitionType
from fastparquet.thrift_structures import parquet_thrift

from jx_base import NESTED, python_type_to_json_type
from mo_dots import concat_field, split_field, join_field
from mo_future import sort_using_key, PY2, text_type
from pyLibrary.env.typed_inserter import json_type_to_inserter_type


class SchemaTree(object):

    def __init__(self):
        self.more = {}  # MAP FROM NAME TO MORE SchemaTree
        self.values = {}  # MAP FROM JSON TYPE TO SchemaElement

    def __getitem__(self, name):
        def _get(node, path):
            if node.more:
                return _get(node.more[path[0]], path[1:])
            else:
                return node.values[path[0]]

        return _get(self, split_field(name))

    @staticmethod
    def new_instance(parquet_schema):
        index = [0]

        def _worker(start):
            output = SchemaTree()
            root = parquet_schema[index[0]]
            if root.type:
                name = join_field(split_field(root.name)[-1:])
                output.values[name] = root
                return output

            max = start + root.num_children
            while index[0] < max:
                name = join_field(split_field(parquet_schema[index[0]].name)[-1:])
                index[0] += 1
                child = _worker(index[0])
                output.more[name] = child
            return output

        return _worker(0).more['.']

    @property
    def leaves(self):
        return [itype for itype in self.values.keys()] + [
                   concat_field(name, leaf)
                   for name, child_schema in self.more.items()
                   for leaf in child_schema.leaves
               ]

    def get_parquet_metadata(
        self,
        path='.'
    ):
        """
        OUTPUT PARQUET METADATA COLUMNS
        :param path: FOR INTERNAL USE
        :return: LIST OF SchemaElement
        """
        children = []
        for name, child_schema in sort_using_key(self.more.items(), lambda p: p[0]):
            children.extend(child_schema.get_parquet_metadata(concat_field(path, name)))
        children.extend(v for k, v in sort_using_key(self.values.items(), lambda p: p[0]))

        return [parquet_thrift.SchemaElement(
            name=path,
            num_children=len(children)
        )] + children

    def max_definition_level(self):
        if not self.more:
            return 1
        else:
            max_child = [m.max_definition_level() for m in self.more.values()]
            return max(max_child) + 1


def get_length(value, dtype):
    if dtype is Type.BYTE_ARRAY:
        return len(value)
    elif dtype is None:
        return 0
    else:
        return dtype.itemsize


def get_repetition_type(jtype):
    return FieldRepetitionType.REPEATED if jtype is NESTED else FieldRepetitionType.OPTIONAL


def merge_schema_element(element, name, value, ptype, dtype, jtype, ittype):
    if not element:
        output = parquet_thrift.SchemaElement(
            name=name,
            type=dtype,
            type_length=get_length(value, dtype),
            repetition_type=get_repetition_type(jtype)
        )
        return output, True
    else:
        element.type_length = max(element.type_length, get_length(value, dtype))

        return element, False



python_type_to_parquet_type = {
    bool: Type.BOOLEAN,
    text_type: Type.BYTE_ARRAY,
    int: Type.INT64,
    float: Type.DOUBLE,
    dict: None,
    list: None
}

if PY2:
    python_type_to_parquet_type[long] = Type.INT64

# MAP FROM PYTHON TYPE TO (parquet_type, json_type, inserter_type)
python_type_to_all_types = {
    ptype: (dtype, python_type_to_json_type[ptype], json_type_to_inserter_type.get(python_type_to_json_type[ptype]))
    for ptype, dtype in python_type_to_parquet_type.items()
}

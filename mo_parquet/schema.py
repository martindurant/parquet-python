# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

from collections import Mapping

from mo_math import MAX
import numpy

from fastparquet.parquet_thrift.parquet.ttypes import ConvertedType, FieldRepetitionType, SchemaElement, Type
from jx_python.jx import count
from mo_dots import Data, coalesce, concat_field, join_field, relative_field, split_field
from mo_future import PY2, none_type, sort_using_key, text_type
from mo_json import NESTED, python_type_to_json_type
from mo_json.typed_encoder import json_type_to_inserter_type
from mo_logs import Log

REQUIRED = FieldRepetitionType.REQUIRED
OPTIONAL = FieldRepetitionType.OPTIONAL
REPEATED = FieldRepetitionType.REPEATED

DEFAULT_RECORD = SchemaElement(name='.', repetition_type=REQUIRED)   # DREMEL ASSUME ALL RECORDS ARE REQUIRED


class SchemaTree(object):

    def __init__(self, locked=False):
        """
        :param locked: DO NOT ALLOW SCHEMA EXPANSION
        """
        self.element = DEFAULT_RECORD
        self.more = {}  # MAP FROM NAME TO MORE SchemaTree
        self.diff_schema = []  # PLACEHOLDER OR NET-NEW COLUMNS ADDED DURING SCHEMA EXPANSION
        self.locked = locked
        self.numpy_type = None

    def add(self, name, repetition_type, type):
        """
        :param name: dot delimited path to the property (use dot (".") for none)
        :param repetition_type: one of OPTIONAL or NESTED (REQUIRED is not possible)
        :param json_type: the json type to store
        :return:
        """
        base_name = self.element.name
        path = split_field(relative_field(name, base_name))
        output = self
        for i, n in enumerate(path[:-1]):
            next = output.more.get(n)
            if next:
                output = next
            else:
                output = output._add_one(concat_field(base_name, join_field(path[0:i + 1])), OPTIONAL, object)
        n = output.more.get(path[-1])
        if n:
            Log.error("can not redefine a property")
        else:
            return output._add_one(name, repetition_type, type)

    def _add_one(self, full_name, repetition_type, type):
        simple_name = split_field(full_name)[-1]
        ntype, ptype, ltype, jtype, itype, length = python_type_to_all_types[type]

        if not isinstance(repetition_type, (list, tuple)):
            repetition_type = [repetition_type]

        last = first = self.more[simple_name] = SchemaTree()
        last.locked = self.locked
        last.numpy_type = ntype

        for rt in repetition_type[:-1]:
            last.element = SchemaElement(
                name='.',
                repetition_type=rt
            )
            temp = last.more['.'] = SchemaTree()
            temp.locked = self.locked
            last = temp

        last.element = SchemaElement(
            name=full_name,
            type=ptype,
            type_length=length,
            repetition_type=repetition_type[-1],
            converted_type=ltype
        )

        return first


    def __getitem__(self, name):
        def _get(node, path):
            if not path:
                return self.element

            return _get(node.more[path[0]], path[1:])

        return _get(self, split_field(name))

    @staticmethod
    def new_instance(parquet_schema_list):
        off = type(str(''), (), dict(set=0))()

        def _worker(start):
            output = SchemaTree()
            root = parquet_schema_list[off.set]

            output.element = root
            max = start + coalesce(root.num_children, 0)

            if off.set == 0:
                if root.name not in ['.', 'schema', 'spark_schema', 'm', 'hive_schema', 'root']:  # some known root names
                    Log.warning("first SchemaElement is given name {{name|quote}}, name is ignored", name=root.name)
                root.name = '.'
                root.repetition_type = REQUIRED

            while off.set < max:
                off.set += 1
                child = _worker(off.set)
                parent = output
                path = relative_field(child.element.name, root.name)

                # path = split_field(relative_field(child.element.name, root.name))
                # for i, p in enumerate(path[:-1]):
                #     new_parent = parent.more[p] = SchemaTree()
                #     new_parent.element = SchemaElement(
                #         name=concat_field(root.name, join_field(path[:i+1])),
                #         repetition_type=REQUIRED
                #     )
                #     parent = new_parent
                # parent.more[path[-1]] = child
                parent.more[path] = child
            return output

        output = _worker(0)
        return output

    @property
    def leaves(self):
        output = set(
            concat_field(name, leaf)
            for name, child_schema in self.more.items()
            for leaf in child_schema.leaves
        )
        if self.element.type is not None:
            output.add('.')

        return output

    def schema_element(self, path):
        if isinstance(path, text_type):
            path = split_field(path)
        output = self
        for p in path:
            output = output.more.get(p)
            if output is None:
                return None
        while '.' in output.more:
            output = output.more['.']
        return output.element

    def _path_to_schema_element(self, path):
        if isinstance(path, text_type):
            def _find(sub_schema):
                if sub_schema.element.name==path:
                    return (sub_schema,)

                for m in sub_schema.more.values():
                    p = _find(m)
                    if p:
                        return (sub_schema, )+p

            return _find(self)
        else:
            output = [self]
            for p in path:
                next = output[-1].more.get(p)
                if next is None:
                    return []
                else:
                    output.append(next)
            while '.' in output[-1].more:
                output.append(output[-1].more['.'])
            return output

    def is_required(self, path):
        return self.schema_element(path).repetition_type == REQUIRED

    def max_definition_level(self, path):
        path = self._path_to_schema_element(path)
        return count(p for p in path if p.element.repetition_type != REQUIRED)

    def max_repetition_level(self, path):
        path = self._path_to_schema_element(path)
        return count(p for p in path if p.element.repetition_type == REPEATED)


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

        if path == '.':
            return children
        else:
            self.element.num_children = len(children)
            return [self.element] + children


def get_length(dtype, value=None):
    if dtype is Type.BYTE_ARRAY:
        if value is None:
            Log.error("we need to guess the length")
        return len(value)
    elif dtype is None:
        return 0
    else:
        return dtype.itemsize


def get_repetition_type(jtype):
    return FieldRepetitionType.REPEATED if jtype is NESTED else FieldRepetitionType.OPTIONAL


def merge_schema_element(element, name, value, ptype, ltype, dtype, jtype, ittype, length):
    element.type_length = MAX((element.type_length, length))
    return element


all_type_to_parquet_type = {
    none_type: None,
    bool: Type.BOOLEAN,
    text_type: Type.BYTE_ARRAY,
    int: Type.INT64,
    float: Type.DOUBLE,
    dict: None,
    object: None,
    Data: None,
    list: None
}


all_type_to_parquet_logical_type = {
    none_type: None,
    bool: None,
    text_type: ConvertedType.UTF8,
    int: ConvertedType.UINT_64,
    float: None,
    dict: None,
    object: None,
    Data: None,
    Mapping: None,
    list: None
}

all_type_to_numpy_type = {
    none_type: None,
    bool: numpy.dtype(bool),
    text_type: numpy.dtype(text_type),
    int: numpy.dtype(int),
    float: numpy.dtype(float),
    dict: None,
    object: None,
    Data: None,
    Mapping: None,
    list: None
}

all_type_to_length = {
    none_type: None,
    bool: 1,
    text_type: None,
    int: 8,
    float: 8,
    dict: None,
    object: None,
    Data: None,
    Mapping: None,
    list: None
}

if PY2:
    all_type_to_numpy_type[long] = numpy.dtype('int64')
    all_type_to_parquet_type[long] = Type.INT64
    all_type_to_parquet_logical_type[long] = ConvertedType.UINT_64
    all_type_to_length[long] = 8


# MAP FROM PYTHON TYPE TO (parquet_type, parquet_logical_type, json_type, inserter_type)
python_type_to_all_types = {
    ptype: (
        all_type_to_numpy_type[ptype],
        dtype,
        all_type_to_parquet_logical_type[ptype],
        python_type_to_json_type[ptype],
        json_type_to_inserter_type.get(python_type_to_json_type[ptype]),
        all_type_to_length[ptype]
    )
    for ptype, dtype in all_type_to_parquet_type.items()
}



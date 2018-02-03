"""Utils for working with the parquet thrift models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .thrift_structures import parquet_thrift


def _is_list_like(helper, column):
    if len(column.meta_data.path_in_schema) < 3:
        return False
    se = helper.schema_element(
        column.meta_data.path_in_schema[:-2])
    ct = se.converted_type
    if ct != parquet_thrift.ConvertedType.LIST:
        return False
    if len(se.children) > 1:
        return False
    se2 = list(se.children.values())[0]
    if len(se2.children) > 1:
        return False
    if se2.repetition_type != parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    se3 = list(se2.children.values())[0]
    if se3.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    return True


def _is_map_like(helper, column):
    if len(column.meta_data.path_in_schema) < 3:
        return False
    se = helper.schema_element(
        column.meta_data.path_in_schema[:-2])
    ct = se.converted_type
    if ct != parquet_thrift.ConvertedType.MAP:
        return False
    if len(se.children) > 1:
        return False
    se2 = list(se.children.values())[0]
    if len(se2.children) != 2:
        return False
    if se2.repetition_type != parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    if set(se2.children) != {'key', 'value'}:
        return False
    se3 = se2.children['key']
    if se3.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
        return False
    se3 = se2.children['value']
    if se3.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
        return False
    return True

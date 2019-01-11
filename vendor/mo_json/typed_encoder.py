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
from datetime import date, datetime, timedelta
from decimal import Decimal
from json.encoder import encode_basestring
import time

from mo_dots import CLASS, Data, DataObject, FlatList, NullType, SLOT, _get, is_data, join_field, split_field
from mo_dots.objects import OBJ
from mo_future import binary_type, generator_types, is_binary, is_text, long, sort_using_key, text_type
from mo_json import BOOLEAN, ESCAPE_DCT, EXISTS, INTEGER, NESTED, NUMBER, STRING, float2json, python_type_to_json_type
from mo_json.encoder import COLON, COMMA, UnicodeBuilder, json_encoder, problem_serializing
from mo_logs import Log
from mo_logs.strings import quote, utf82unicode
from mo_times import Date, Duration


def encode_property(name):
    return name.replace(",", "\\,").replace(".", ",")


def decode_property(encoded):
    return encoded.replace("\\,", "\a").replace(",", ".").replace("\a", ",")


def untype_path(encoded):
    if encoded.startswith(".."):
        remainder = encoded.lstrip(".")
        back = len(encoded) - len(remainder) - 1
        return ("." * back) + join_field(decode_property(c) for c in split_field(remainder) if not c.startswith(TYPE_PREFIX))
    else:
        return join_field(decode_property(c) for c in split_field(encoded) if not c.startswith(TYPE_PREFIX))


def unnest_path(encoded):
    if encoded.startswith(".."):
        remainder = encoded.lstrip(".")
        back = len(encoded) - len(remainder)
        return ("." * back) + unnest_path(remainder)

    path = split_field(encoded)
    if not path:
        return "."
    if path[-1] == NESTED_TYPE:
        path = path[:-1]
        if not path:
            return "."

    return join_field([decode_property(c) for c in path[:-1] if not c.startswith(TYPE_PREFIX)] + [decode_property(path[-1])])


def untyped(value):
    return _untype_value(value)


def _untype_list(value):
    if any(is_data(v) for v in value):
        # MAY BE MORE TYPED OBJECTS IN THIS LIST
        output = [_untype_value(v) for v in value]
    else:
        # LIST OF PRIMITIVE VALUES
        output = value

    if len(output) == 0:
        return None
    elif len(output) == 1:
        return output[0]
    else:
        return output


def _untype_dict(value):
    output = {}

    for k, v in value.items():
        if k.startswith(TYPE_PREFIX):
            if k == EXISTS_TYPE:
                continue
            elif k == NESTED_TYPE:
                return _untype_list(v)
            else:
                return v
        else:
            new_v = _untype_value(v)
            if new_v is not None:
                output[decode_property(k)] = new_v
    return output


def _untype_value(value):
    _type = _get(value, CLASS)
    if _type is Data:
        return _untype_dict(_get(value, SLOT))
    elif _type is dict:
        return _untype_dict(value)
    elif _type is FlatList:
        return _untype_list(value.list)
    elif _type is list:
        return _untype_list(value)
    elif _type is NullType:
        return None
    elif _type is DataObject:
        return _untype_value(_get(value, OBJ))
    elif _type in generator_types:
        return _untype_list(value)
    else:
        return value


def encode(value):
    buffer = UnicodeBuilder(1024)
    typed_encode(
        value,
        sub_schema={},
        path=[],
        net_new_properties=[],
        buffer=buffer
    )
    return buffer.build()


def typed_encode(value, sub_schema, path, net_new_properties, buffer):
    """
    :param value: THE DATA STRUCTURE TO ENCODE
    :param sub_schema: dict FROM PATH TO Column DESCRIBING THE TYPE
    :param path: list OF CURRENT PATH
    :param net_new_properties: list FOR ADDING NEW PROPERTIES NOT FOUND IN sub_schema
    :param buffer: UnicodeBuilder OBJECT
    :return:
    """
    try:
        # from jx_base import Column
        if sub_schema.__class__.__name__=='Column':
            value_json_type = python_type_to_json_type[value.__class__]
            column_json_type = es_type_to_json_type[sub_schema.es_type]

            if value_json_type == column_json_type:
                pass  # ok
            elif value_json_type == NESTED and all(python_type_to_json_type[v.__class__] == column_json_type for v in value if v != None):
                pass  # empty arrays can be anything
            else:
                from mo_logs import Log

                Log.error("Can not store {{value}} in {{column|quote}}", value=value, column=sub_schema.name)

            sub_schema = {json_type_to_inserter_type[value_json_type]: sub_schema}

        if value == None:
            from mo_logs import Log
            Log.error("can not encode null (missing) values")
        elif value is True:
            if BOOLEAN_TYPE not in sub_schema:
                sub_schema[BOOLEAN_TYPE] = {}
                net_new_properties.append(path + [BOOLEAN_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_BOOLEAN_TYPE)
            append(buffer, 'true}')
            return
        elif value is False:
            if BOOLEAN_TYPE not in sub_schema:
                sub_schema[BOOLEAN_TYPE] = {}
                net_new_properties.append(path + [BOOLEAN_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_BOOLEAN_TYPE)
            append(buffer, 'false}')
            return

        _type = value.__class__
        if _type in (dict, Data):
            if sub_schema.__class__.__name__ == 'Column':
                from mo_logs import Log
                Log.error("Can not handle {{column|json}}", column=sub_schema)

            if NESTED_TYPE in sub_schema:
                # PREFER NESTED, WHEN SEEN BEFORE
                if value:
                    append(buffer, '{')
                    append(buffer, QUOTED_NESTED_TYPE)
                    append(buffer, '[')
                    _dict2json(value, sub_schema[NESTED_TYPE], path + [NESTED_TYPE], net_new_properties, buffer)
                    append(buffer, ']' + COMMA)
                    append(buffer, QUOTED_EXISTS_TYPE)
                    append(buffer, text_type(len(value)))
                    append(buffer, '}')
                else:
                    # SINGLETON LIST
                    append(buffer, '{')
                    append(buffer, QUOTED_NESTED_TYPE)
                    append(buffer, '[{')
                    append(buffer, QUOTED_EXISTS_TYPE)
                    append(buffer, '1}]')
                    append(buffer, COMMA)
                    append(buffer, QUOTED_EXISTS_TYPE)
                    append(buffer, '1}')
            else:
                if EXISTS_TYPE not in sub_schema:
                    sub_schema[EXISTS_TYPE] = {}
                    net_new_properties.append(path + [EXISTS_TYPE])

                if value:
                    _dict2json(value, sub_schema, path, net_new_properties, buffer)
                else:
                    append(buffer, '{')
                    append(buffer, QUOTED_EXISTS_TYPE)
                    append(buffer, '1}')
        elif _type is binary_type:
            if STRING_TYPE not in sub_schema:
                sub_schema[STRING_TYPE] = True
                net_new_properties.append(path + [STRING_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_STRING_TYPE)
            append(buffer, '"')
            try:
                v = utf82unicode(value)
            except Exception as e:
                raise problem_serializing(value, e)

            for c in v:
                append(buffer, ESCAPE_DCT.get(c, c))
            append(buffer, '"}')
        elif _type is text_type:
            if STRING_TYPE not in sub_schema:
                sub_schema[STRING_TYPE] = True
                net_new_properties.append(path + [STRING_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_STRING_TYPE)
            append(buffer, '"')
            for c in value:
                append(buffer, ESCAPE_DCT.get(c, c))
            append(buffer, '"}')
        elif _type in (int, long):
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])

            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, text_type(value))
            append(buffer, '}')
        elif _type in (float, Decimal):
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, float2json(value))
            append(buffer, '}')
        elif _type in (set, list, tuple, FlatList):
            if len(value) == 0:
                append(buffer, '{')
                append(buffer, QUOTED_EXISTS_TYPE)
                append(buffer, '0}')
            elif any(v.__class__ in (Data, dict, set, list, tuple, FlatList) for v in value):
                # THIS IS NOT DONE BECAUSE
                if len(value) == 1:
                    if NESTED_TYPE in sub_schema:
                        append(buffer, '{')
                        append(buffer, QUOTED_NESTED_TYPE)
                        _list2json(value, sub_schema[NESTED_TYPE], path + [NESTED_TYPE], net_new_properties, buffer)
                        append(buffer, '}')
                    else:
                        # NO NEED TO NEST, SO DO NOT DO IT
                        typed_encode(value[0], sub_schema, path, net_new_properties, buffer)
                else:
                    if NESTED_TYPE not in sub_schema:
                        sub_schema[NESTED_TYPE] = {}
                        net_new_properties.append(path + [NESTED_TYPE])
                    append(buffer, '{')
                    append(buffer, QUOTED_NESTED_TYPE)
                    _list2json(value, sub_schema[NESTED_TYPE], path + [NESTED_TYPE], net_new_properties, buffer)
                    append(buffer, '}')
            else:
                # ALLOW PRIMITIVE MULTIVALUES
                value = [v for v in value if v != None]
                types = list(set(json_type_to_inserter_type[python_type_to_json_type[v.__class__]] for v in value))
                if len(types) == 0:  # HANDLE LISTS WITH Nones IN THEM
                    append(buffer, '{')
                    append(buffer, QUOTED_NESTED_TYPE)
                    append(buffer, '[]}')
                elif len(types) > 1:
                    _list2json(value, sub_schema, path + [NESTED_TYPE], net_new_properties, buffer)
                else:
                    element_type = types[0]
                    if element_type not in sub_schema:
                        sub_schema[element_type] = True
                        net_new_properties.append(path + [element_type])
                    append(buffer, '{')
                    append(buffer, quote(element_type))
                    append(buffer, COLON)
                    _multivalue2json(value, sub_schema[element_type], path + [element_type], net_new_properties, buffer)
                    append(buffer, '}')
        elif _type is date:
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, float2json(time.mktime(value.timetuple())))
            append(buffer, '}')
        elif _type is datetime:
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, float2json(time.mktime(value.timetuple())))
            append(buffer, '}')
        elif _type is Date:
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, float2json(value.unix))
            append(buffer, '}')
        elif _type is timedelta:
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, float2json(value.total_seconds()))
            append(buffer, '}')
        elif _type is Duration:
            if NUMBER_TYPE not in sub_schema:
                sub_schema[NUMBER_TYPE] = True
                net_new_properties.append(path + [NUMBER_TYPE])
            append(buffer, '{')
            append(buffer, QUOTED_NUMBER_TYPE)
            append(buffer, float2json(value.seconds))
            append(buffer, '}')
        elif _type is NullType:
            append(buffer, 'null')
        elif hasattr(value, '__data__'):
            typed_encode(value.__data__(), sub_schema, path, net_new_properties, buffer)
        elif hasattr(value, '__iter__'):
            if NESTED_TYPE not in sub_schema:
                sub_schema[NESTED_TYPE] = {}
                net_new_properties.append(path + [NESTED_TYPE])

            append(buffer, '{')
            append(buffer, QUOTED_NESTED_TYPE)
            _iter2json(value, sub_schema[NESTED_TYPE], path + [NESTED_TYPE], net_new_properties, buffer)
            append(buffer, '}')
        else:
            from mo_logs import Log

            Log.error(text_type(repr(value)) + " is not JSON serializable")
    except Exception as e:
        from mo_logs import Log

        Log.error(text_type(repr(value)) + " is not JSON serializable", cause=e)


def _list2json(value, sub_schema, path, net_new_properties, buffer):
    if not value:
        append(buffer, '[]')
    else:
        sep = '['
        for v in value:
            append(buffer, sep)
            sep = COMMA
            typed_encode(v, sub_schema, path, net_new_properties, buffer)
        append(buffer, ']')
        append(buffer, COMMA)
        append(buffer, QUOTED_EXISTS_TYPE)
        append(buffer, text_type(len(value)))


def _multivalue2json(value, sub_schema, path, net_new_properties, buffer):
    if not value:
        append(buffer, '[]')
    elif len(value) == 1:
        append(buffer, json_encoder(value[0]))
    else:
        sep = '['
        for v in value:
            append(buffer, sep)
            sep = COMMA
            append(buffer, json_encoder(v))
        append(buffer, ']')


def _iter2json(value, sub_schema, path, net_new_properties, buffer):
    append(buffer, '[')
    sep = ''
    count = 0
    for v in value:
        append(buffer, sep)
        sep = COMMA
        typed_encode(v, sub_schema, path, net_new_properties, buffer)
        count += 1
    append(buffer, ']')
    append(buffer, COMMA)
    append(buffer, QUOTED_EXISTS_TYPE)
    append(buffer, text_type(count))


def _dict2json(value, sub_schema, path, net_new_properties, buffer):
    prefix = '{'
    for k, v in sort_using_key(value.items(), lambda r: r[0]):
        if v == None or v == '':
            continue
        append(buffer, prefix)
        prefix = COMMA
        if is_binary(k):
            k = utf82unicode(k)
        if not is_text(k):
            Log.error("Expecting property name to be a string")
        if k not in sub_schema:
            sub_schema[k] = {}
            net_new_properties.append(path + [k])
        append(buffer, encode_basestring(encode_property(k)))
        append(buffer, COLON)
        typed_encode(v, sub_schema[k], path + [k], net_new_properties, buffer)
    if prefix is COMMA:
        append(buffer, COMMA)
        append(buffer, QUOTED_EXISTS_TYPE)
        append(buffer, '1}')
    else:
        append(buffer, '{')
        append(buffer, QUOTED_EXISTS_TYPE)
        append(buffer, '1}')




TYPE_PREFIX = "~"  # u'\u0442\u0443\u0440\u0435-'  # "туре"
BOOLEAN_TYPE = TYPE_PREFIX + "b~"
NUMBER_TYPE = TYPE_PREFIX + "n~"
STRING_TYPE = TYPE_PREFIX + "s~"
NESTED_TYPE = TYPE_PREFIX + "N~"
EXISTS_TYPE = TYPE_PREFIX + "e~"

append = UnicodeBuilder.append

QUOTED_BOOLEAN_TYPE = quote(BOOLEAN_TYPE) + COLON
QUOTED_NUMBER_TYPE = quote(NUMBER_TYPE) + COLON
QUOTED_STRING_TYPE = quote(STRING_TYPE) + COLON
QUOTED_NESTED_TYPE = quote(NESTED_TYPE) + COLON
QUOTED_EXISTS_TYPE = quote(EXISTS_TYPE) + COLON

inserter_type_to_json_type = {
    BOOLEAN_TYPE: BOOLEAN,
    NUMBER_TYPE: NUMBER,
    STRING_TYPE: STRING
}

json_type_to_inserter_type = {
    BOOLEAN: BOOLEAN_TYPE,
    INTEGER: NUMBER_TYPE,
    NUMBER: NUMBER_TYPE,
    STRING: STRING_TYPE,
    NESTED: NESTED_TYPE,
    EXISTS: EXISTS_TYPE
}

es_type_to_json_type = {
    "text": "string",
    "string": "string",
    "keyword": "string",
    "float": "number",
    "double": "number",
    "integer": "number",
    "object": "object",
    "nested": "nested",
    "source": "json",
    "boolean": "boolean",
    "exists": "exists"
}

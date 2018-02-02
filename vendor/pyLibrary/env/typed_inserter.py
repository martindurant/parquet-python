# encoding: utf-8
#
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

import json
import time
from collections import Mapping
from datetime import datetime, date, timedelta
from decimal import Decimal

from mo_future import text_type, binary_type
from jx_python.meta import Column

from jx_base import python_type_to_json_type, INTEGER, NUMBER, EXISTS, NESTED, STRING, BOOLEAN, STRUCT, OBJECT
from mo_dots import Data, FlatList, NullType, unwrap
from mo_future import utf8_json_encoder, long
from mo_json import ESCAPE_DCT, float2json, json2value
from mo_json.encoder import pretty_json, problem_serializing, UnicodeBuilder, COMMA, COLON
from mo_json.typed_encoder import untype_path, encode_property, BOOLEAN_TYPE, NESTED_TYPE, EXISTS_TYPE, STRING_TYPE, NUMBER_TYPE, TYPE_PREFIX
from mo_logs import Log
from mo_logs.strings import utf82unicode, quote
from mo_times.dates import Date
from mo_times.durations import Duration
from pyLibrary.env.elasticsearch import parse_properties, random_id, es_type_to_json_type

append = UnicodeBuilder.append


QUOTED_BOOLEAN_TYPE = quote(BOOLEAN_TYPE)
QUOTED_NUMBER_TYPE = quote(NUMBER_TYPE)
QUOTED_STRING_TYPE = quote(STRING_TYPE)
QUOTED_NESTED_TYPE = quote(NESTED_TYPE)
QUOTED_EXISTS_TYPE = quote(EXISTS_TYPE)

json_type_to_inserter_type = {
    BOOLEAN: BOOLEAN_TYPE,
    INTEGER: NUMBER_TYPE,
    NUMBER: NUMBER_TYPE,
    STRING: STRING_TYPE,
    NESTED: NESTED_TYPE,
    EXISTS: EXISTS_TYPE
}



class TypedInserter(object):
    def __init__(self, es=None, id_column="_id"):
        self.es = es
        self.id_column = id_column
        self.remove_id = True if id_column == "_id" else False

        if es:
            _schema = Data()
            for c in parse_properties(es.settings.alias, ".", es.get_properties()):
                if c.type not in (OBJECT, NESTED):
                    _schema[c.names["."]] = c
            self.schema = unwrap(_schema)
        else:
            self.schema = {}

    def typed_encode(self, r):
        """
        :param record:  expecting id and value properties
        :return:  dict with id and json properties
        """
        try:
            value = r['value']
            if "json" in r:
                value = json2value(r["json"])
            elif isinstance(value, Mapping) or value != None:
                pass
            else:
                from mo_logs import Log
                raise Log.error("Expecting every record given to have \"value\" or \"json\" property")

            _buffer = UnicodeBuilder(1024)
            net_new_properties = []
            path = []
            if isinstance(value, Mapping):
                given_id = value.get(self.id_column)
                if self.remove_id:
                    value[self.id_column] = None
            else:
                given_id = None

            if given_id:
                record_id = r.get('id')
                if record_id and record_id != given_id:
                    from mo_logs import Log

                    raise Log.error(
                        "expecting {{property}} of record ({{record_id|quote}}) to match one given ({{given|quote}})",
                        property=self.id_column,
                        record_id=record_id,
                        given=given_id
                    )
            else:
                record_id = r.get('id')
                if record_id:
                    given_id = record_id
                else:
                    given_id = random_id()

            self._typed_encode(value, self.schema, path, net_new_properties, _buffer)
            json = _buffer.build()

            for props in net_new_properties:
                path, type = props[:-1], props[-1][1:]
                # self.es.add_column(join_field(path), type)

            return {"id": given_id, "json": json}
        except Exception as e:
            # THE PRETTY JSON WILL PROVIDE MORE DETAIL ABOUT THE SERIALIZATION CONCERNS
            from mo_logs import Log

            Log.error("Serialization of JSON problems", cause=e)

    def _typed_encode(self, value, sub_schema, path, net_new_properties, _buffer):
        try:
            if isinstance(sub_schema, Column):
                value_json_type = python_type_to_json_type[value.__class__]
                column_json_type = es_type_to_json_type[sub_schema.type]

                if value_json_type == column_json_type:
                    pass  # ok
                else:
                    from mo_logs import Log

                    Log.error("Can not store {{value}} in {{column|quote}}", value=value, column=sub_schema.names['.'])

                sub_schema = {json_type_to_inserter_type[value_json_type]: sub_schema}

            if value is None:
                append(_buffer, '{}')
                return
            elif value is True:
                if BOOLEAN_TYPE not in sub_schema:
                    sub_schema[BOOLEAN_TYPE] = {}
                    net_new_properties.append(path+[BOOLEAN_TYPE])
                append(_buffer, '{'+QUOTED_BOOLEAN_TYPE+COLON+'true}')
                return
            elif value is False:
                if BOOLEAN_TYPE not in sub_schema:
                    sub_schema[BOOLEAN_TYPE] = {}
                    net_new_properties.append(path+[BOOLEAN_TYPE])
                append(_buffer, '{'+QUOTED_BOOLEAN_TYPE+COLON+'false}')
                return

            _type = value.__class__
            if _type in (dict, Data):
                if isinstance(sub_schema, Column):
                    from mo_logs import Log
                    Log.error("Can not handle {{column|json}}", column=sub_schema)

                if NESTED_TYPE in sub_schema:
                    # PREFER NESTED, WHEN SEEN BEFORE
                    if value:
                        append(_buffer, '{'+QUOTED_NESTED_TYPE+COLON+'[')
                        self._dict2json(value, sub_schema[NESTED_TYPE], path + [NESTED_TYPE], net_new_properties, _buffer)
                        append(_buffer, ']'+COMMA+QUOTED_EXISTS_TYPE+COLON + text_type(len(value)) + '}')
                    else:
                        # SINGLETON LISTS OF null SHOULD NOT EXIST
                        from mo_logs import Log

                        Log.error("should not happen")
                else:
                    if EXISTS_TYPE not in sub_schema:
                        sub_schema[EXISTS_TYPE] = {}
                        net_new_properties.append(path+[EXISTS_TYPE])

                    if value:
                        self._dict2json(value, sub_schema, path, net_new_properties, _buffer)
                    else:
                        append(_buffer, '{'+QUOTED_EXISTS_TYPE+COLON+'0}')
            elif _type is binary_type:
                if STRING_TYPE not in sub_schema:
                    sub_schema[STRING_TYPE] = True
                    net_new_properties.append(path + [STRING_TYPE])
                append(_buffer, '{'+QUOTED_STRING_TYPE+COLON+'"')
                try:
                    v = utf82unicode(value)
                except Exception as e:
                    raise problem_serializing(value, e)

                for c in v:
                    append(_buffer, ESCAPE_DCT.get(c, c))
                append(_buffer, '"}')
            elif _type is text_type:
                if STRING_TYPE not in sub_schema:
                    sub_schema[STRING_TYPE] = True
                    net_new_properties.append(path + [STRING_TYPE])

                append(_buffer, '{'+QUOTED_STRING_TYPE+COLON+'"')
                for c in value:
                    append(_buffer, ESCAPE_DCT.get(c, c))
                append(_buffer, '"}')
            elif _type in (int, long, Decimal):
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])

                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(value))
                append(_buffer, '}')
            elif _type is float:
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])
                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(value))
                append(_buffer, '}')
            elif _type in (set, list, tuple, FlatList):
                if len(value) == 0:
                    append(_buffer, '{'+QUOTED_NESTED_TYPE+COLON+'[]}')
                elif any(isinstance(v, (Mapping, set, list, tuple, FlatList)) for v in value):
                    if NESTED_TYPE not in sub_schema:
                        sub_schema[NESTED_TYPE] = {}
                        net_new_properties.append(path + [NESTED_TYPE])
                    append(_buffer, '{'+QUOTED_NESTED_TYPE+COLON)
                    self._list2json(value, sub_schema[NESTED_TYPE], path+[NESTED_TYPE], net_new_properties, _buffer)
                    append(_buffer, '}')
                else:
                    # ALLOW PRIMITIVE MULTIVALUES
                    types = list(set(python_type_to_json_type[v.__class__] for v in value))
                    if len(types) > 1:
                        from mo_logs import Log
                        Log.error("Can not handle multi-typed multivalues")
                    element_type = json_type_to_inserter_type[types[0]]
                    if element_type not in sub_schema:
                        sub_schema[element_type] = True
                        net_new_properties.append(path + [element_type])
                    append(_buffer, '{'+quote(element_type)+COLON)
                    self._multivalue2json(value, sub_schema[element_type], path+[element_type], net_new_properties, _buffer)
                    append(_buffer, '}')
            elif _type is date:
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])
                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(time.mktime(value.timetuple())))
                append(_buffer, '}')
            elif _type is datetime:
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])
                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(time.mktime(value.timetuple())))
                append(_buffer, '}')
            elif _type is Date:
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])
                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(value.unix))
                append(_buffer, '}')
            elif _type is timedelta:
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])
                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(value.total_seconds()))
                append(_buffer, '}')
            elif _type is Duration:
                if NUMBER_TYPE not in sub_schema:
                    sub_schema[NUMBER_TYPE] = True
                    net_new_properties.append(path + [NUMBER_TYPE])
                append(_buffer, '{'+QUOTED_NUMBER_TYPE+COLON)
                append(_buffer, float2json(value.seconds))
                append(_buffer, '}')
            elif _type is NullType:
                append(_buffer, 'null')
            elif hasattr(value, '__json__'):
                from mo_logs import Log
                Log.error("do not know how to handle")
            elif hasattr(value, '__iter__'):
                if NESTED_TYPE not in sub_schema:
                    sub_schema[NESTED_TYPE] = {}
                    net_new_properties.append(path + [NESTED_TYPE])

                append(_buffer, '{'+QUOTED_NESTED_TYPE+COLON)
                self._iter2json(value, sub_schema[NESTED_TYPE], path+[NESTED_TYPE], net_new_properties, _buffer)
                append(_buffer, '}')
            else:
                from mo_logs import Log

                Log.error(text_type(repr(value)) + " is not JSON serializable")
        except Exception as e:
            from mo_logs import Log

            Log.error(text_type(repr(value)) + " is not JSON serializable", cause=e)

    def _list2json(self, value, sub_schema, path, net_new_properties, _buffer):
        if not value:
            append(_buffer, '[]')
        else:
            sep = '['
            for v in value:
                append(_buffer, sep)
                sep = COMMA
                self._typed_encode(v, sub_schema, path, net_new_properties, _buffer)
            append(_buffer, ']'+COMMA+QUOTED_EXISTS_TYPE+COLON+text_type(len(value)))

    def _multivalue2json(self, value, sub_schema, path, net_new_properties, _buffer):
        if not value:
            append(_buffer, '[]')
        elif len(value) == 1:
            append(_buffer, json_encoder(value[0]))
        else:
            sep = '['
            for v in value:
                append(_buffer, sep)
                sep = COMMA
                append(_buffer, json_encoder(v))
            append(_buffer, ']')

    def _iter2json(self, value, sub_schema, path, net_new_properties, _buffer):
        append(_buffer, '[')
        sep = ''
        count = 0
        for v in value:
            append(_buffer, sep)
            sep = COMMA
            self._typed_encode(v, sub_schema, path, net_new_properties, _buffer)
            count += 1
        append(_buffer, ']'+COMMA+QUOTED_EXISTS_TYPE+COLON+ + text_type(count))

    def _dict2json(self, value, sub_schema, path, net_new_properties, _buffer):
        prefix = '{'
        for k, v in ((kk, value[kk]) for kk in sorted(value.keys())):
            if v == None or v == '':
                continue
            append(_buffer, prefix)
            prefix = COMMA
            if isinstance(k, binary_type):
                k = utf82unicode(k)
            if not isinstance(k, text_type):
                Log.error("Expecting property name to be a string")
            if k not in sub_schema:
                sub_schema[k] = {}
                net_new_properties.append(path+[k])
            append(_buffer, json.dumps(encode_property(k)))
            append(_buffer, COLON)
            self._typed_encode(v, sub_schema[k], path+[k], net_new_properties, _buffer)
        if prefix == COMMA:
            append(_buffer, COMMA+QUOTED_EXISTS_TYPE+COLON+'1}')
        else:
            append(_buffer, '{'+QUOTED_EXISTS_TYPE+COLON+'0}')


json_encoder = utf8_json_encoder

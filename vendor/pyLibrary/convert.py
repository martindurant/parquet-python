# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, absolute_import, division, unicode_literals

import ast
import base64
import cgi
import datetime
from decimal import Decimal
import gzip
import hashlib
from io import BytesIO
import json
import re
from tempfile import TemporaryFile

from mo_dots import concat_field, unwrap, unwraplist, wrap
from mo_future import HTMLParser, PY3, StringIO, is_binary, is_text, long, text_type
import mo_json
from mo_logs import Log
from mo_logs.exceptions import suppress_exception
from mo_logs.strings import expand_template, quote
import mo_math
from mo_times.dates import Date

"""
DUE TO MY POOR MEMORY, THIS IS A LIST OF ALL CONVERSION ROUTINES
IN <from_type> "2" <to_type> FORMAT
"""

value2json = mo_json.value2json
json2value = mo_json.json2value


def string2datetime(value, format=None):
    return unix2datetime(Date(value, format).unix)


def string2boolean(value):
    if value in ["true", "T"]:
        return True
    elif value in ["false", "F"]:
        return False
    else:
        return None


_v2b = {
    True: True,
    "true": True,
    "T": True,
    1: True,
    False: False,
    "false": False,
    "F": False,
    0: False,
    None: None
}


def value2boolean(value):
    return _v2b.get(value, True)


def str2datetime(value, format=None):
    return unix2datetime(Date(value, format).unix)


def datetime2string(value, format="%Y-%m-%d %H:%M:%S"):
    try:
        return value.strftime(format)
    except Exception as e:
        from mo_logs import Log

        Log.error("Can not format {{value}} with {{format}}", value=value, format=format, cause=e)


def datetime2str(value, format="%Y-%m-%d %H:%M:%S"):
    return Date(value).format(format=format)


def datetime2unix(d):
    try:
        if d == None:
            return None
        elif isinstance(d, datetime.datetime):
            epoch = datetime.datetime(1970, 1, 1)
        elif isinstance(d, datetime.date):
            epoch = datetime.date(1970, 1, 1)
        else:
            Log.error("Can not convert {{value}} of type {{type}}",  value= d,  type= d.__class__)

        diff = d - epoch
        return Decimal(long(diff.total_seconds() * 1000000)) / 1000000
    except Exception as e:
        Log.error("Can not convert {{value}}", value=d, cause=e)


def datetime2milli(d):
    return datetime2unix(d) * 1000


def timedelta2milli(v):
    return v.total_seconds()


def unix2datetime(u):
    try:
        if u == None:
            return None
        if u == 9999999999: # PYPY BUG https://bugs.pypy.org/issue1697
            return datetime.datetime(2286, 11, 20, 17, 46, 39)
        return datetime.datetime.utcfromtimestamp(u)
    except Exception as e:
        Log.error("Can not convert {{value}} to datetime",  value= u, cause=e)


def milli2datetime(u):
    if u == None:
        return None
    return unix2datetime(u / 1000.0)


def dict2Multiset(dic):
    if dic == None:
        return None
    from mo_collections.multiset import Multiset
    output = Multiset()
    output.dic = unwrap(dic).copy()
    return output


def multiset2dict(value):
    """
    CONVERT MULTISET TO dict THAT MAPS KEYS TO MAPS KEYS TO KEY-COUNT
    """
    if value == None:
        return None
    return dict(value.dic)


def table2list(
    column_names, # tuple of columns names
    rows          # list of tuples
):
    return wrap([dict(zip(column_names, r)) for r in rows])

def table2tab(
    column_names, # tuple of columns names
    rows          # list of tuples
):
    def row(r):
        return "\t".join(map(value2json, r))

    return row(column_names)+"\n"+("\n".join(row(r) for r in rows))



def list2tab(rows):
    columns = set()
    for r in wrap(rows):
        columns |= set(k for k, v in r.leaves())
    keys = list(columns)

    output = []
    for r in wrap(rows):
        output.append("\t".join(value2json(r[k]) for k in keys))

    return "\t".join(keys) + "\n" + "\n".join(output)


def list2table(rows, column_names=None):
    if column_names:
        keys = list(set(column_names))
    else:
        columns = set()
        for r in rows:
            columns |= set(r.keys())
        keys = list(columns)

    output = [[unwraplist(r.get(k)) for k in keys] for r in rows]

    return wrap({
        "meta": {"format": "table"},
        "header": keys,
        "data": output
    })


def list2cube(rows, column_names=None):
    if column_names:
        keys = column_names
    else:
        columns = set()
        for r in rows:
            columns |= set(r.keys())
        keys = list(columns)

    data = {k: [] for k in keys}
    output = wrap({
        "meta": {"format": "cube"},
        "edges": [
            {
                "name": "rownum",
                "domain": {"type": "rownum", "min": 0, "max": len(rows), "interval": 1}
            }
        ],
        "data": data
    })

    for r in rows:
        for k in keys:
            data[k].append(unwraplist(r[k]))

    return output


def value2string(value):
    # PROPER NULL HANDLING
    if value == None:
        return None
    return text_type(value)


def value2quote(value):
    # RETURN PRETTY PYTHON CODE FOR THE SAME
    if is_text(value):
        return string2quote(value)
    else:
        return text_type(repr(value))


def string2quote(value):
    if value == None:
        return "None"
    return quote(value)


string2regexp = re.escape


def string2url(value):
    if is_text(value):
        return "".join([_map2url[c] for c in unicode2latin1(value)])
    elif is_binary(value):
        return "".join([_map2url[c] for c in value])
    else:
        Log.error("Expecting a string")


# def url_param2value(param):
#     """
#     CONVERT URL QUERY PARAMETERS INTO DICT
#     """
#     if is_text(param):
#         param = param.encode("ascii")
#
#     def _decode(v):
#         output = []
#         i = 0
#         while i < len(v):
#             c = v[i]
#             if c == "%":
#                 d = hex2bytes(v[i + 1:i + 3])
#                 output.append(d)
#                 i += 3
#             else:
#                 output.append(c)
#                 i += 1
#
#         output = (b"".join(output)).decode("latin1")
#         try:
#             return json2value(output)
#         except Exception:
#             pass
#         return output
#
#
#     query = {}
#     for p in param.split(b'&'):
#         if not p:
#             continue
#         if p.find(b"=") == -1:
#             k = p
#             v = True
#         else:
#             k, v = p.split(b"=")
#             v = _decode(v)
#
#         u = query.get(k)
#         if u is None:
#             query[k] = v
#         elif is_list(u):
#             u += [v]
#         else:
#             query[k] = [u, v]
#
#     return query


def html2unicode(value):
    # http://stackoverflow.com/questions/57708/convert-xml-html-entities-into-unicode-string-in-python
    return HTMLParser.HTMLParser().unescape(value)


def unicode2html(value):
    return cgi.escape(value)


def unicode2latin1(value):
    output = value.encode("latin1")
    return output


def quote2string(value):
    with suppress_exception:
        return ast.literal_eval(value)


# RETURN PYTHON CODE FOR THE SAME

def value2code(value):
    return text_type(repr(value))


def DataFrame2string(df, columns=None):
    output = StringIO.StringIO()
    try:
        df.to_csv(output, sep="\t", header=True, cols=columns, engine='python')
        return output.getvalue()
    finally:
        output.close()


def ascii2char(ascii):
    return chr(ascii)


def char2ascii(char):
    return ord(char)


def ascii2unicode(value):
    return value.decode("latin1")


def latin12hex(value):
    return value.encode("hex")


def int2hex(value, size):
    return (("0" * size) + hex(value)[2:])[-size:]


if PY3:
    def hex2bytes(value):
        return bytearray.fromhex(value)
else:
    def hex2bytes(value):
        return value.decode("hex")


def bytes2hex(value, separator=" "):
    return separator.join("%02X" % ord(x) for x in value)


def base642bytearray(value):
    if value == None:
        return bytearray(b"")
    else:
        return bytearray(base64.b64decode(value))


def base642bytes(value):
    return base64.b64decode(value)


def bytes2base64(value):
    if isinstance(value, bytearray):
        value=str(value)
    return base64.b64encode(value).decode("utf8")


def bytes2sha1(value):
    if is_text(value):
        Log.error("can not convert unicode to sha1")
    sha = hashlib.sha1(value)
    return sha.hexdigest()


def value2intlist(value):
    if value == None:
        return []
    elif hasattr(value, '__iter__'):
        output = [int(d) for d in value if d != "" and d != None]
        return output
    elif isinstance(value, int):
        return [value]
    elif value.strip() == "":
        return []
    else:
        return [int(value)]

def value2int(value):
    if value == None:
        return None
    else:
        return int(value)


def value2number(v):
    try:
        if isinstance(v, float) and round(v, 0) != v:
            return v
            # IF LOOKS LIKE AN INT, RETURN AN INT
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception as e:
            Log.error("Not a number ({{value}})",  value= v, cause=e)


def latin12unicode(value):
    if is_text(value):
        Log.error("can not convert unicode from latin1")
    try:
        return text_type(value.decode('latin1'))
    except Exception as e:
        Log.error("Can not convert {{value|quote}} to unicode", value=value)


def pipe2value(value):
    type = value[0]
    if type == '0':
        return None
    if type == 'n':
        return value2number(value[1::])

    if type != 's' and type != 'a':
        Log.error("unknown pipe type ({{type}}) in {{value}}",  type= type,  value= value)

    # EXPECTING MOST STRINGS TO NOT HAVE ESCAPED CHARS
    output = _unPipe(value)
    if type == 's':
        return output

    return [pipe2value(v) for v in output.split("|")]


def zip2bytes(compressed):
    """
    UNZIP DATA
    """
    if hasattr(compressed, "read"):
        return gzip.GzipFile(fileobj=compressed, mode='r')

    buff = BytesIO(compressed)
    archive = gzip.GzipFile(fileobj=buff, mode='r')
    from pyLibrary.env.big_data import safe_size
    return safe_size(archive)


def bytes2zip(bytes):
    """
    RETURN COMPRESSED BYTES
    """
    if hasattr(bytes, "read"):
        buff = TemporaryFile()
        archive = gzip.GzipFile(fileobj=buff, mode='w')
        for b in bytes:
            archive.write(b)
        archive.close()
        buff.seek(0)
        from pyLibrary.env.big_data import FileString, safe_size
        return FileString(buff)

    buff = BytesIO()
    archive = gzip.GzipFile(fileobj=buff, mode='w')
    archive.write(bytes)
    archive.close()
    return buff.getvalue()


def ini2value(ini_content):
    """
    INI FILE CONTENT TO Data
    """
    from mo_future import ConfigParser, StringIO

    buff = StringIO(ini_content)
    config = ConfigParser()
    config._read(buff, "dummy")

    output = {}
    for section in config.sections():
        output[section]=s = {}
        for k, v in config.items(section):
            s[k]=v
    return wrap(output)


if PY3:
    _map2url = {chr(i).encode('latin1'): chr(i) for i in range(32, 256)}
    for c in " {}<>;/?:@&=+$,":
        _map2url[c] = "%" + int2hex(ord(c), 2)
else:
    _map2url = {chr(i): chr(i).decode('latin1') for i in range(32, 256)}
    for c in " {}<>;/?:@&=+$,":
        _map2url[c] = "%" + int2hex(ord(c), 2)



def _unPipe(value):
    s = value.find("\\", 1)
    if s < 0:
        return value[1::]

    result = ""
    e = 1
    while True:
        c = value[s + 1]
        if c == 'p':
            result = result + value[e:s] + '|'
            s += 2
            e = s
        elif c == '\\':
            result = result + value[e:s] + '\\'
            s += 2
            e = s
        else:
            s += 1

        s = value.find("\\", s)
        if s < 0:
            break
    return result + value[e::]

json_decoder = json.JSONDecoder().decode


def json_schema_to_markdown(schema):
    from jx_python import jx

    def _md_code(code):
        return "`"+code+"`"

    def _md_italic(value):
        return "*"+value+"*"

    def _inner(schema, parent_name, indent):
        more_lines = []
        for k, v in schema.items():
            full_name = concat_field(parent_name, k)
            details = indent+"* "+_md_code(full_name)
            if v.type:
                details += " - "+_md_italic(v.type)
            else:
                Log.error("{{full_name}} is missing type", full_name=full_name)
            if v.description:
                details += " " + v.description
            more_lines.append(details)

            if v.type in ["object", "array", "nested"]:
                more_lines.extend(_inner(v.properties, full_name, indent+"  "))
        return more_lines

    lines = []
    if schema.title:
        lines.append("# "+schema.title)

    lines.append(schema.description)
    lines.append("")

    for k, v in jx.sort(schema.properties.items(), 0):
        full_name = k
        if v.type in ["object", "array", "nested"]:
            lines.append("## "+_md_code(full_name)+" Property")
            if v.description:
                lines.append(v.description)
            lines.append("")

            if v.type in ["object", "array", "nested"]:
                lines.extend(_inner(v.properties, full_name, "  "))
        else:
            lines.append("## "+_md_code(full_name)+" ("+v.type+")")
            if v.description:
                lines.append(v.description)

    return "\n".join(lines)


def table2csv(table_data):
    """
    :param table_data: expecting a list of tuples
    :return: text in nice formatted csv
    """
    text_data = [tuple(value2json(vals, pretty=True) for vals in rows) for rows in table_data]

    col_widths = [max(len(text) for text in cols) for cols in zip(*text_data)]
    template = ", ".join(
        "{{" + text_type(i) + "|left_align(" + text_type(w) + ")}}"
        for i, w in enumerate(col_widths)
    )
    text = "\n".join(expand_template(template, d) for d in text_data)
    return text

ZeroMoment2dict = mo_math.stats.ZeroMoment2dict




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

import cgi
import json as _json
import math
import re
import string
from collections import Mapping
from datetime import datetime as builtin_datetime
from datetime import timedelta, date
from json.encoder import encode_basestring

from mo_dots import coalesce, wrap, get_module
from mo_future import text_type, xrange, binary_type, round as _round, PY3
from mo_logs.convert import datetime2unix, datetime2string, value2json, milli2datetime, unix2datetime
from mo_logs.url import value2url_param

_json_encoder = None
_Log = None
_Except = None
_Duration = None


def _late_import():
    global _json_encoder
    global _Log
    global _Except
    global _Duration

    try:
        _json_encoder = get_module("mo_json.encoder").json_encoder
    except Exception:
        _json_encoder = _json.dumps
    from mo_logs import Log as _Log
    from mo_logs.exceptions import Except as _Except
    from mo_times.durations import Duration as _Duration

    _ = _json_encoder
    _ = _Log
    _ = _Except
    _ = _Duration


def expand_template(template, value):
    """
    :param template: A UNICODE STRING WITH VARIABLE NAMES IN MOUSTACHES `{{}}`
    :param value: Data HOLDING THE PARAMTER VALUES
    :return: UNICODE STRING WITH VARIABLES EXPANDED
    """
    value = wrap(value)
    if isinstance(template, text_type):
        return _simple_expand(template, (value,))

    return _expand(template, (value,))


def datetime(value):
    if isinstance(value, (date, builtin_datetime)):
        pass
    elif value < 10000000000:
        value = unix2datetime(value)
    else:
        value = milli2datetime(value)

    return datetime2string(value, "%Y-%m-%d %H:%M:%S")


def unicode(value):
    if value == None:
        return ""
    return text_type(value)


def unix(value):
    if isinstance(value, (date, builtin_datetime)):
        pass
    elif value < 10000000000:
        value = unix2datetime(value)
    else:
        value = milli2datetime(value)

    return str(datetime2unix(value))


def url(value):
    """
    convert FROM dict OR string TO URL PARAMETERS
    """
    return value2url_param(value)


def html(value):
    """
    convert FROM unicode TO HTML OF THE SAME
    """
    return cgi.escape(value)


def upper(value):
    return value.upper()


def lower(value):
    return value.lower()


def newline(value):
    """
    ADD NEWLINE, IF SOMETHING
    """
    return "\n" + toString(value).lstrip("\n")


def replace(value, find, replace):
    return value.replace(find, replace)


def json(value, pretty=True):
    if not _Duration:
        _late_import()
    return _json_encoder(value, pretty=pretty)


def tab(value):
    if isinstance(value, Mapping):
        h, d = zip(*wrap(value).leaves())
        return \
            "\t".join(map(value2json, h)) + \
            "\n" + \
            "\t".join(map(value2json, d))
    else:
        text_type(value)


def indent(value, prefix=u"\t", indent=None):
    if indent != None:
        prefix = prefix * indent

    value = toString(value)
    try:
        content = value.rstrip()
        suffix = value[len(content):]
        lines = content.splitlines()
        return prefix + (u"\n" + prefix).join(lines) + suffix
    except Exception as e:
        raise Exception(u"Problem with indent of value (" + e.message + u")\n" + text_type(toString(value)))


def outdent(value):
    try:
        num = 100
        lines = toString(value).splitlines()
        for l in lines:
            trim = len(l.lstrip())
            if trim > 0:
                num = min(num, len(l) - len(l.lstrip()))
        return u"\n".join([l[num:] for l in lines])
    except Exception as e:
        if not _Log:
            _late_import()

        _Log.error("can not outdent value", e)


def round(value, decimal=None, digits=None, places=None):
    """
    :param value:  THE VALUE TO ROUND
    :param decimal: NUMBER OF DECIMAL PLACES TO ROUND (NEGATIVE IS LEFT-OF-DECIMAL)
    :param digits: ROUND TO SIGNIFICANT NUMBER OF digits
    :param places: SAME AS digits
    :return:
    """
    value = float(value)
    if value == 0.0:
        return "0"

    digits = coalesce(digits, places)
    if digits != None:
        left_of_decimal = int(math.ceil(math.log10(abs(value))))
        decimal = digits - left_of_decimal

    right_of_decimal = max(decimal, 0)
    format = "{:." + text_type(right_of_decimal) + "f}"
    return format.format(_round(value, decimal))


def percent(value, decimal=None, digits=None, places=None):
    value = float(value)
    if value == 0.0:
        return "0%"

    digits = coalesce(digits, places)
    if digits != None:
        left_of_decimal = int(math.ceil(math.log10(abs(value)))) + 2
        decimal = digits - left_of_decimal

    decimal = coalesce(decimal, 0)
    right_of_decimal = max(decimal, 0)
    format = "{:." + text_type(right_of_decimal) + "%}"
    return format.format(_round(value, decimal + 2))


def find(value, find, start=0):
    """
    MUCH MORE USEFUL VERSION OF string.find()
    """
    l = len(value)
    if isinstance(find, list):
        m = l
        for f in find:
            i = value.find(f, start)
            if i == -1:
                continue
            m = min(m, i)
        return m
    else:
        i = value.find(find, start)
        if i == -1:
            return l
        return i


def strip(value):
    """
    REMOVE WHITESPACE (INCLUDING CONTROL CHARACTERS)
    """
    if not value or (ord(value[0]) > 32 and ord(value[-1]) > 32):
        return value

    s = 0
    e = len(value)
    while s < e:
        if ord(value[s]) > 32:
            break
        s += 1
    else:
        return ""

    for i in reversed(range(s, e)):
        if ord(value[i]) > 32:
            return value[s:i + 1]

    return ""


def trim(value):
    return strip(value)


def between(value, prefix, suffix, start=0):
    value = toString(value)
    if prefix == None:
        e = value.find(suffix, start)
        if e == -1:
            return None
        else:
            return value[:e]

    s = value.find(prefix, start)
    if s == -1:
        return None
    s += len(prefix)

    e = value.find(suffix, s)
    if e == -1:
        return None

    s = value.rfind(prefix, start, e) + len(prefix)  # WE KNOW THIS EXISTS, BUT THERE MAY BE A RIGHT-MORE ONE

    return value[s:e]


def right(value, len):
    if len <= 0:
        return u""
    return value[-len:]


def right_align(value, length):
    if length <= 0:
        return u""

    value = text_type(value)

    if len(value) < length:
        return (" " * (length - len(value))) + value
    else:
        return value[-length:]


def left_align(value, length):
    if length <= 0:
        return u""

    value = text_type(value)

    if len(value) < length:
        return value + (" " * (length - len(value)))
    else:
        return value[:length]


def left(value, len):
    if len <= 0:
        return u""
    return value[0:len]


def comma(value):
    """
    FORMAT WITH THOUSANDS COMMA (,) SEPARATOR
    """
    try:
        if float(value) == _round(float(value), 0):
            output = "{:,}".format(int(value))
        else:
            output = "{:,}".format(float(value))
    except Exception:
        output = text_type(value)

    return output


def quote(value):
    if value == None:
        output = ""
    elif isinstance(value, text_type):
        output = encode_basestring(value)
    else:
        output = _json.dumps(value)
    return output


def hex(value):
    return hex(value)



_SNIP = "...<snip>..."


def limit(value, length):
    # LIMIT THE STRING value TO GIVEN LENGTH
    if len(value) <= length:
        return value
    elif length < len(_SNIP) * 2:
        return value[0:length]
    else:
        lhs = int(round((length - len(_SNIP)) / 2, 0))
        rhs = length - len(_SNIP) - lhs
        return value[:lhs] + _SNIP + value[-rhs:]


def split(value, sep="\n"):
    # GENERATOR VERSION OF split()
    # SOMETHING TERRIBLE HAPPENS, SOMETIMES, IN PYPY
    s = 0
    len_sep = len(sep)
    n = value.find(sep, s)
    while n > -1:
        yield value[s:n]
        s = n + len_sep
        n = value.find(sep, s)
    yield value[s:]
    value = None


def common_prefix(*args):
    prefix = args[0]
    for a in args[1:]:
        for i in range(min(len(prefix), len(a))):
            if a[i] != prefix[i]:
                prefix = prefix[:i]
                break
    return prefix


def find_first(value, find_arr, start=0):
    i = len(value)
    for f in find_arr:
        temp = value.find(f, start)
        if temp == -1: continue
        i = min(i, temp)
    if i == len(value): return -1
    return i


def is_hex(value):
    return all(c in string.hexdigits for c in value)


pattern = re.compile(r"\{\{([\w_\.]+(\|[^\}^\|]+)*)\}\}")


def _expand(template, seq):
    """
    seq IS TUPLE OF OBJECTS IN PATH ORDER INTO THE DATA TREE
    """
    if isinstance(template, text_type):
        return _simple_expand(template, seq)
    elif isinstance(template, Mapping):
        template = wrap(template)
        assert template["from"], "Expecting template to have 'from' attribute"
        assert template.template, "Expecting template to have 'template' attribute"

        data = seq[-1][template["from"]]
        output = []
        for d in data:
            s = seq + (d,)
            output.append(_expand(template.template, s))
        return coalesce(template.separator, "").join(output)
    elif isinstance(template, list):
        return "".join(_expand(t, seq) for t in template)
    else:
        if not _Log:
            _late_import()

        _Log.error("can not handle")


def _simple_expand(template, seq):
    """
    seq IS TUPLE OF OBJECTS IN PATH ORDER INTO THE DATA TREE
    seq[-1] IS THE CURRENT CONTEXT
    """

    def replacer(found):
        ops = found.group(1).split("|")

        path = ops[0]
        var = path.lstrip(".")
        depth = min(len(seq), max(1, len(path) - len(var)))
        try:
            val = seq[-depth]
            if var:
                if isinstance(val, (list, tuple)) and float(var) == _round(float(var), 0):
                    val = val[int(var)]
                else:
                    val = val[var]
            for func_name in ops[1:]:
                parts = func_name.split('(')
                if len(parts) > 1:
                    val = eval(parts[0] + "(val, " + ("(".join(parts[1::])))
                else:
                    val = globals()[func_name](val)
            val = toString(val)
            return val
        except Exception as e:
            from mo_logs import Except

            e = Except.wrap(e)
            try:
                if e.message.find("is not JSON serializable"):
                    # WORK HARDER
                    val = toString(val)
                    return val
            except Exception as f:
                if not _Log:
                    _late_import()

                _Log.warning(
                    "Can not expand " + "|".join(ops) + " in template: {{template_|json}}",
                    template_=template,
                    cause=e
                )
            return "[template expansion error: (" + str(e.message) + ")]"

    return pattern.sub(replacer, template)


if PY3:
    delchars = "".join(c for c in map(chr, range(256)) if not c.isalnum())
else:
    delchars = "".join(c.decode("latin1") for c in map(chr, range(256)) if not c.decode("latin1").isalnum())


def deformat(value):
    """
    REMOVE NON-ALPHANUMERIC CHARACTERS

    FOR SOME REASON translate CAN NOT BE CALLED:
        ERROR: translate() takes exactly one argument (2 given)
        File "C:\Python27\lib\string.py", line 493, in translate
    """
    output = []
    for c in value:
        if c in delchars:
            continue
        output.append(c)
    return "".join(output)


def toString(val):
    if not _Duration:
        _late_import()

    if val == None:
        return ""
    elif isinstance(val, (Mapping, list, set)):
        return _json_encoder(val, pretty=True)
    elif hasattr(val, "__json__"):
        return val.__json__()
    elif isinstance(val, _Duration):
        return text_type(round(val.seconds, places=4)) + " seconds"
    elif isinstance(val, timedelta):
        duration = val.total_seconds()
        return text_type(round(duration, 3)) + " seconds"
    elif isinstance(val, text_type):
        return val
    elif isinstance(val, str):
        try:
            return val.decode('utf8')
        except Exception as _:
            pass

        try:
            return val.decode('latin1')
        except Exception as e:
            if not _Log:
                _late_import()

            _Log.error(text_type(type(val)) + " type can not be converted to unicode", cause=e)
    else:
        try:
            return text_type(val)
        except Exception as e:
            if not _Log:
                _late_import()

            _Log.error(text_type(type(val)) + " type can not be converted to unicode", cause=e)


def edit_distance(s1, s2):
    """
    FROM http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance# Python
    LICENCE http://creativecommons.org/licenses/by-sa/3.0/
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return 1.0

    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1] / len(s1)


DIFF_PREFIX = re.compile(r"@@ -(\d+(?:\s*,\d+)?) \+(\d+(?:\s*,\d+)?) @@")


def apply_diff(text, diff, reverse=False):
    """
    SOME EXAMPLES OF diff
    #@@ -1 +1 @@
    #-before china goes live, the content team will have to manually update the settings for the china-ready apps currently in marketplace.
    #+before china goes live (end January developer release, June general audience release) , the content team will have to manually update the settings for the china-ready apps currently in marketplace.
    @@ -0,0 +1,3 @@
    +before china goes live, the content team will have to manually update the settings for the china-ready apps currently in marketplace.
    +
    +kward has the details.
    @@ -1 +1 @@
    -before china goes live (end January developer release, June general audience release), the content team will have to manually update the settings for the china-ready apps currently in marketplace.
    +before china goes live , the content team will have to manually update the settings for the china-ready apps currently in marketplace.
    @@ -3 +3 ,6 @@
    -kward has the details.+kward has the details.
    +
    +Target Release Dates :
    +https://mana.mozilla.org/wiki/display/PM/Firefox+OS+Wave+Launch+Cross+Functional+View
    +
    +Content Team Engagement & Tasks : https://appreview.etherpad.mozilla.org/40
    """
    if not diff:
        return text
    if diff[0].strip() == "":
        return text

    matches = DIFF_PREFIX.match(diff[0].strip())
    if not matches:
        if not _Log:
            _late_import()

        _Log.error("Can not handle {{diff}}\n",  diff= diff[0])

    remove = [int(i.strip()) for i in matches.group(1).split(",")]
    if len(remove) == 1:
        remove = [remove[0], 1]  # DEFAULT 1
    add = [int(i.strip()) for i in matches.group(2).split(",")]
    if len(add) == 1:
        add = [add[0], 1]

    # UNUSUAL CASE WHERE @@ -x +x, n @@ AND FIRST LINE HAS NOT CHANGED
    half = int(len(diff[1]) / 2)
    first_half = diff[1][:half]
    last_half = diff[1][half:half * 2]
    if remove[1] == 1 and add[0] == remove[0] and first_half[1:] == last_half[1:]:
        diff[1] = first_half
        diff.insert(2, last_half)

    if not reverse:
        if remove[1] != 0:
            text = text[:remove[0] - 1] + text[remove[0] + remove[1] - 1:]
        text = text[:add[0] - 1] + [d[1:] for d in diff[1 + remove[1]:1 + remove[1] + add[1]]] + text[add[0] - 1:]
        text = apply_diff(text, diff[add[1] + remove[1] + 1:], reverse=reverse)
    else:
        text = apply_diff(text, diff[add[1] + remove[1] + 1:], reverse=reverse)
        if add[1] != 0:
            text = text[:add[0] - 1] + text[add[0] + add[1] - 1:]
        text = text[:remove[0] - 1] + [d[1:] for d in diff[1:1 + remove[1]]] + text[remove[0] - 1:]

    return text


def utf82unicode(value):
    """
    WITH EXPLANATION FOR FAILURE
    """
    try:
        return value.decode("utf8")
    except Exception as e:
        if not _Log:
            _late_import()

        if not isinstance(value, binary_type):
            _Log.error("Can not convert {{type}} to unicode because it's not bytes",  type= type(value).__name__)

        e = _Except.wrap(e)
        for i, c in enumerate(value):
            try:
                c.decode("utf8")
            except Exception as f:
                _Log.error("Can not convert charcode {{c}} in string  index {{i}}", i=i, c=ord(c), cause=[e, _Except.wrap(f)])

        try:
            latin1 = text_type(value.decode("latin1"))
            _Log.error("Can not explain conversion failure, but seems to be latin1", e)
        except Exception:
            pass

        try:
            a = text_type(value.decode("latin1"))
            _Log.error("Can not explain conversion failure, but seems to be latin1", e)
        except Exception:
            pass

        _Log.error("Can not explain conversion failure of " + type(value).__name__ + "!", e)


def wordify(value):
    return [w for w in re.split(r"[\W_]", value) if strip(w)]




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

import math
import re
from datetime import datetime, date, timedelta
from decimal import Decimal
from time import time as _time

from mo_future import text_type, PY3, long

from mo_dots import Null
from mo_logs import Except
from mo_logs.strings import deformat

from mo_times.durations import Duration, MILLI_VALUES
from mo_times.vendor.dateutil.parser import parse as parse_date

_utcnow = datetime.utcnow

try:
    import pytz
except Exception:
    pass

ISO8601 = '%Y-%m-%dT%H:%M:%SZ'


class Date(object):
    __slots__ = ["unix"]

    MIN = None
    MAX = None

    def __new__(cls, *args, **kwargs):
        if not args or (len(args) == 1 and args[0] == None):
            return Null
        return parse(*args)

    def __init__(self, *args):
        if self.unix is None:
            self.unix = parse(*args).unix

    def __nonzero__(self):
        return True

    def floor(self, duration=None):
        if duration is None:  # ASSUME DAY
            return _unix2Date(math.floor(self.unix / 86400) * 86400)
        elif duration.month:
            dt = unix2datetime(self.unix)
            month = int(math.floor((dt.year*12+dt.month-1) / duration.month) * duration.month)
            year = int(math.floor(month/12))
            month -= 12*year
            return Date(datetime(year, month+1, 1))
        elif duration.milli % (7 * 86400000) == 0:
            offset = 4*86400
            return _unix2Date(math.floor((self.unix + offset) / duration.seconds) * duration.seconds - offset)
        else:
            return _unix2Date(math.floor(self.unix / duration.seconds) * duration.seconds)

    def format(self, format="%Y-%m-%d %H:%M:%S"):
        try:
            return unix2datetime(self.unix).strftime(format)
        except Exception as e:
            from mo_logs import Log

            Log.error("Can not format {{value}} with {{format}}", value=unix2datetime(self.unix), format=format, cause=e)

    @property
    def milli(self):
        return self.unix*1000

    @property
    def hour(self):
        """
        :return: HOUR (int) IN THE GMT DAY
        """
        return int(int(self.unix)/60/60 % 24)

    def addDay(self):
        return Date(unix2datetime(self.unix) + timedelta(days=1))

    def add(self, other):
        if other==None:
            return Null
        elif isinstance(other, (datetime, date)):
            return _unix2Date(self.unix - datetime2unix(other))
        elif isinstance(other, Date):
            return _unix2Date(self.unix - other.unix)
        elif isinstance(other, timedelta):
            return Date(unix2datetime(self.unix) + other)
        elif isinstance(other, Duration):
            if other.month:
                value = unix2datetime(self.unix)
                if (value+timedelta(days=1)).month != value.month:
                    # LAST DAY OF MONTH
                    output = add_month(value+timedelta(days=1), other.month) - timedelta(days=1)
                    return Date(output)
                else:
                    day = value.day
                    num_days = (add_month(datetime(value.year, value.month, 1), other.month+1) - timedelta(days=1)).day
                    day = min(day, num_days)
                    curr = set_day(value, day)
                    output = add_month(curr, other.month)
                    return Date(output)
            else:
                return _unix2Date(self.unix + other.seconds)
        else:
            from mo_logs import Log

            Log.error("can not subtract {{type}} from Date", type=other.__class__.__name__)

    @staticmethod
    def now():
        return _unix2Date(_time())

    @staticmethod
    def eod():
        """
        RETURN END-OF-TODAY (WHICH IS SAME AS BEGINNING OF TOMORROW)
        """
        return _unix2Date(Date.today().unix + 86400)

    @staticmethod
    def today():
        return _unix2Date(math.floor(_time() / 86400) * 86400)

    @staticmethod
    def range(min, max, interval):
        v = min
        while v < max:
            yield v
            v = v + interval

    def __str__(self):
        return str(unix2datetime(self.unix))

    def __repr__(self):
        return unix2datetime(self.unix).__repr__()

    def __sub__(self, other):
        if other == None:
            return None
        if isinstance(other, datetime):
            return Duration(self.unix - Date(other).unix)
        if isinstance(other, Date):
            return Duration(self.unix - other.unix)

        return self.add(-other)

    def __lt__(self, other):
        other = Date(other)
        return self.unix < other.unix

    def __eq__(self, other):
        if other == None:
            return Null

        try:
            return other.unix == self.unix
        except Exception:
            pass

        try:
            return Date(other).unix == self.unix
        except Exception:
            return False

    def __le__(self, other):
        other = Date(other)
        return self.unix <= other.unix

    def __gt__(self, other):
        other = Date(other)
        return self.unix > other.unix

    def __ge__(self, other):
        other = Date(other)
        return self.unix >= other.unix

    def __add__(self, other):
        return self.add(other)

    def __data__(self):
        return self.unix

    @classmethod
    def min(cls, *values):
        output = Null
        for v in values:
            if output == None and v != None:
                output = v
            elif v < output:
                output = v
        return output


def parse(*args):
    try:
        if len(args) == 1:
            a0 = args[0]
            if isinstance(a0, (datetime, date)):
                output = _unix2Date(datetime2unix(a0))
            elif isinstance(a0, Date):
                output = _unix2Date(a0.unix)
            elif isinstance(a0, (int, long, float, Decimal)):
                a0 = float(a0)
                if a0 > 9999999999:    # WAY TOO BIG IF IT WAS A UNIX TIMESTAMP
                    output = _unix2Date(a0 / 1000)
                else:
                    output = _unix2Date(a0)
            elif isinstance(a0, text_type) and len(a0) in [9, 10, 12, 13] and is_integer(a0):
                a0 = float(a0)
                if a0 > 9999999999:    # WAY TOO BIG IF IT WAS A UNIX TIMESTAMP
                    output = _unix2Date(a0 / 1000)
                else:
                    output = _unix2Date(a0)
            elif isinstance(a0, text_type):
                output = unicode2Date(a0)
            else:
                output = _unix2Date(datetime2unix(datetime(*args)))
        else:
            if isinstance(args[0], text_type):
                output = unicode2Date(*args)
            else:
                output = _unix2Date(datetime2unix(datetime(*args)))

        return output
    except Exception as e:
        from mo_logs import Log

        Log.error("Can not convert {{args}} to Date", args=args, cause=e)


def add_month(offset, months):
    month = int(offset.month+months-1)
    year = offset.year
    if not 0 <= month < 12:
        r = _mod(month, 12)
        year += int((month - r) / 12)
        month = r
    month += 1

    output = datetime(
        year=year,
        month=month,
        day=offset.day,
        hour=offset.hour,
        minute=offset.minute,
        second=offset.second,
        microsecond=offset.microsecond
    )
    return output


def set_day(offset, day):
    output = datetime(
        year=offset.year,
        month=offset.month,
        day=day,
        hour=offset.hour,
        minute=offset.minute,
        second=offset.second,
        microsecond=offset.microsecond
    )
    return output


def parse_time_expression(value):
    def simple_date(sign, dig, type, floor):
        if dig or sign:
            from mo_logs import Log
            Log.error("can not accept a multiplier on a datetime")

        if floor:
            return Date(type).floor(Duration(floor))
        else:
            return Date(type)

    terms = re.match(r'(\d*[|\w]+)\s*([+-]\s*\d*[|\w]+)*', value).groups()

    sign, dig, type = re.match(r'([+-]?)\s*(\d*)([|\w]+)', terms[0]).groups()
    if "|" in type:
        type, floor = type.split("|")
    else:
        floor = None

    if type in MILLI_VALUES.keys():
        value = Duration(dig+type)
    else:
        value = simple_date(sign, dig, type, floor)

    for term in terms[1:]:
        if not term:
            continue
        sign, dig, type = re.match(r'([+-])\s*(\d*)([|\w]+)', term).groups()
        if "|" in type:
            type, floor = type.split("|")
        else:
            floor = None

        op = {"+": "__add__", "-": "__sub__"}[sign]
        if type in MILLI_VALUES.keys():
            if floor:
                from mo_logs import Log
                Log.error("floor (|) of duration not accepted")
            value = value.__getattribute__(op)(Duration(dig+type))
        else:
            value = value.__getattribute__(op)(simple_date(sign, dig, type, floor))

    return value


def unicode2Date(value, format=None):
    """
    CONVERT UNICODE STRING TO UNIX TIMESTAMP VALUE
    """
    ## http://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
    if value == None:
        return None

    if format != None:
        try:
            if format.endswith("%S.%f") and "." not in value:
                value += ".000"
            return _unix2Date(datetime2unix(datetime.strptime(value, format)))
        except Exception as e:
            from mo_logs import Log

            Log.error("Can not format {{value}} with {{format}}", value=value, format=format, cause=e)

    value = value.strip()
    if value.lower() == "now":
        return _unix2Date(datetime2unix(_utcnow()))
    elif value.lower() == "today":
        return _unix2Date(math.floor(datetime2unix(_utcnow()) / 86400) * 86400)
    elif value.lower() in ["eod", "tomorrow"]:
        return _unix2Date(math.floor(datetime2unix(_utcnow()) / 86400) * 86400 + 86400)

    if any(value.lower().find(n) >= 0 for n in ["now", "today", "eod", "tomorrow"] + list(MILLI_VALUES.keys())):
        return parse_time_expression(value)

    try:  # 2.7 DOES NOT SUPPORT %z
        local_value = parse_date(value)  #eg 2014-07-16 10:57 +0200
        return _unix2Date(datetime2unix((local_value - local_value.utcoffset()).replace(tzinfo=None)))
    except Exception as e:
        e = Except.wrap(e)  # FOR DEBUGGING
        pass

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f"
    ]
    for f in formats:
        try:
            return _unix2Date(datetime2unix(datetime.strptime(value, f)))
        except Exception:
            pass



    deformats = [
        "%Y-%m",# eg 2014-07-16 10:57 +0200
        "%Y%m%d",
        "%d%m%Y",
        "%d%m%y",
        "%d%b%Y",
        "%d%b%y",
        "%d%B%Y",
        "%d%B%y",
        "%Y%m%d%H%M%S",
        "%Y%m%dT%H%M%S",
        "%d%m%Y%H%M%S",
        "%d%m%y%H%M%S",
        "%d%b%Y%H%M%S",
        "%d%b%y%H%M%S",
        "%d%B%Y%H%M%S",
        "%d%B%y%H%M%S"
    ]
    value = deformat(value)
    for f in deformats:
        try:
            return unicode2Date(value, format=f)
        except Exception:
            pass

    else:
        from mo_logs import Log
        Log.error("Can not interpret {{value}} as a datetime",  value= value)


DATETIME_EPOCH = datetime(1970, 1, 1)
DATE_EPOCH = date(1970, 1, 1)


def datetime2unix(value):
    try:
        if value == None:
            return None
        elif isinstance(value, datetime):
            diff = value - DATETIME_EPOCH
            return diff.total_seconds()
        elif isinstance(value, date):
            diff = value - DATE_EPOCH
            return diff.total_seconds()
        else:
            from mo_logs import Log
            Log.error("Can not convert {{value}} of type {{type}}", value=value, type=value.__class__)
    except Exception as e:
        from mo_logs import Log
        Log.error("Can not convert {{value}}", value=value, cause=e)


def unix2datetime(unix):
    return datetime.utcfromtimestamp(unix)


def unix2Date(unix):
    if not isinstance(unix, float):
        from mo_logs import Log
        Log.error("problem")
    return _unix2Date(unix)


def _unix2Date(unix):
    output = object.__new__(Date)
    output.unix = unix
    return output


if PY3:
    delchars = "".join(c for c in map(chr, range(256)) if not c.isalnum())
else:
    delchars = "".join(c.decode("latin1") for c in map(chr, range(256)) if not c.decode("latin1").isalnum())


def deformat(value):
    """
    REMOVE NON-ALPHANUMERIC CHARACTERS

    FOR SOME REASON translate CAN NOT BE CALLED:
        ERROR: translate() takes exactly one argument (2 given)
        File "C:\\Python27\\lib\\string.py", line 493, in translate
    """
    output = []
    for c in value:
        if c in delchars:
            continue
        output.append(c)
    return "".join(output)


Date.MIN = Date(datetime(1, 1, 1))
Date.MAX = Date(datetime(2286, 11, 20, 17, 46, 39))


def is_integer(s):
    if s is True or s is False:
        return False

    try:
        if float(s) == round(float(s), 0):
            return True
        return False
    except Exception:
        return False


def _mod(value, mod=1):
    """
    RETURN NON-NEGATIVE MODULO
    RETURN None WHEN GIVEN INVALID ARGUMENTS
    """
    if value == None:
        return None
    elif mod <= 0:
        return None
    elif value < 0:
        return (value % mod + mod) % mod
    else:
        return value % mod

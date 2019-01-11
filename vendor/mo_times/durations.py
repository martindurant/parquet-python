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

import datetime
import re

from mo_dots import get_module, wrap
from mo_future import is_text, text_type
from mo_math import MIN, is_nan, is_number, abs, floor, round
from mo_times.vendor.dateutil.relativedelta import relativedelta

_Date = None
_Log = None


def _delayed_import():
    global _Date
    global _Log

    from mo_times.dates import Date as _Date
    from mo_logs import Log as _Log

    _ = _Date
    _ = _Log


class Duration(object):
    __slots__ = ["_milli", "month"]

    def __new__(cls, value=None, **kwargs):
        output = object.__new__(cls)
        if value == None:
            if kwargs:
                output.milli = datetime.timedelta(**kwargs).total_seconds() * 1000
                output.month = 0
                return output
            else:
                return None

        if is_number(value):
            output._milli = float(value) * 1000
            output.month = 0
            return output
        elif is_text(value):
            return parse(value)
        elif isinstance(value, Duration):
            output.milli = value.milli
            output.month = value.month
            return output
        elif isinstance(value, float) and is_nan(value):
            return None
        else:
            from mo_logs import Log
            Log.error("Do not know type of object (" + get_module("mo_json").value2json(value) + ")of to make a Duration")

    @staticmethod
    def range(start, stop, step):
        if not step:
            if not _Log:
                _delayed_import()
            _Log.error("Expecting a non-zero duration for interval")
        output = []
        c = start
        while c < stop:
            output.append(c)
            c += step
        return output



    def __add__(self, other):
        output = Duration(0)
        output.milli = self.milli + other.milli
        output.month = self.month + other.month
        return output

    def __radd__(self, other):
        if not _Date:
            _delayed_import()

        if other == None:
            return None
        if isinstance(other, datetime.datetime):
            return _Date(other).add(self)
        elif isinstance(other, _Date):
            return other.add(self)
        return self + other

    def __mul__(self, amount):
        output = Duration(0)
        output.milli = self.milli * amount
        output.month = self.month * amount
        return output

    def __neg__(self):
        output = Duration(0)
        output.milli = -self.milli
        output.month = -self.month
        return output


    def __rmul__(self, amount):
        amount = float(amount)
        output = Duration(0)
        output.milli = self.milli * amount
        output.month = self.month * amount
        return output

    def __div__(self, amount):
        if isinstance(amount, Duration) and amount.month:
            m = self.month
            r = self.milli

            # DO NOT CONSIDER TIME OF DAY
            tod = r % MILLI_VALUES.day
            r = r - tod

            if m == 0 and r > (MILLI_VALUES.year / 3):
                m = floor(12 * self.milli / MILLI_VALUES.year)
                r -= (m / 12) * MILLI_VALUES.year
            else:
                r = r - (self.month * MILLI_VALUES.month)
                if r >= MILLI_VALUES.day * 31:
                    from mo_logs import Log
                    Log.error("Do not know how to handle")
            r = MIN([29 / 30, (r + tod) / (MILLI_VALUES.day * 30)])

            output = floor(m / amount.month) + r
            return output
        elif is_number(amount):
            output = Duration(0)
            output.milli = self.milli / amount
            output.month = self.month / amount
            return output
        else:
            return self.milli / amount.milli

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __sub__(self, duration):
        output = Duration(0)
        output.milli = self.milli - duration.milli
        output.month = self.month - duration.month
        return output

    def __rsub__(self, time):
        if isinstance(time, Duration):
            output = Duration(0)
            output.milli = time.milli - self.milli
            output.month = time.month - self.month
            return output
        else:
            return time - relativedelta(months=self.month, seconds=self.milli/1000)

    def __lt__(self, other):
        if other == None:
            return False
        return self.milli < Duration(other).milli

    def __le__(self, other):
        if other == None:
            return False
        return self.milli <= Duration(other).milli

    def __ge__(self, other):
        if other == None:
            return True
        return self.milli >= Duration(other).milli

    def __gt__(self, other):
        if other == None:
            return True
        return self.milli > Duration(other).milli

    def ceiling(self, interval=None):
        if interval is None:
            interval = DAY
        return (self + interval).floor(interval) - interval

    def floor(self, interval=None):
        if not isinstance(interval, Duration):
            from mo_logs import Log
            Log.error("Expecting an interval as a Duration object")

        output = Duration(0)
        if interval.month:
            if self.month:
                output.month = int(floor(self.month / interval.month) * interval.month)
                output.milli = output.month * MILLI_VALUES.month
                return output

            # A MONTH OF DURATION IS BIGGER THAN A CANONICAL MONTH
            output.month = int(floor(self.milli * 12 / MILLI_VALUES["year"] / interval.month) * interval.month)
            output.milli = output.month * MILLI_VALUES.month
        else:
            output.milli = floor(self.milli / (interval.milli)) * (interval.milli)
        return output

    @property
    def seconds(self):
        return self._milli / 1000

    @property
    def milli(self):
        return self._milli

    @milli.setter
    def milli(self, value):
        if not isinstance(value, float):
            from mo_logs import Log
            Log.error("not allowed")
        self._milli = value

    def total_seconds(self):
        return float(self.milli) / 1000

    def __str__(self):
        return str(self.__unicode__())

    def __unicode__(self):
        if not self.milli:
            return "zero"

        output = ""
        rest = (self.milli - (MILLI_VALUES.month * self.month)) # DO NOT INCLUDE THE MONTH'S MILLIS
        isNegative = (rest < 0)
        rest = abs(rest)

        # MILLI
        rem = rest % 1000
        if rem != 0:
            output = "+" + text_type(rem) + "milli" + output
        rest = floor(rest / 1000)

        # SECOND
        rem = rest % 60
        if rem != 0:
            output = "+" + text_type(rem) + "second" + output
        rest = floor(rest / 60)

        # MINUTE
        rem = rest % 60
        if rem != 0:
            output = "+" + text_type(rem) + "minute" + output
        rest = floor(rest / 60)

        # HOUR
        rem = rest % 24
        if rem != 0:
            output = "+" + text_type(rem) + "hour" + output
        rest = floor(rest / 24)

        # DAY
        if (rest < 11 and rest != 7) or rest % 10 == 0:
            rem = rest
            rest = 0
        else:
            rem = rest % 7
            rest = floor(rest / 7)

        if rem != 0:
            output = "+" + text_type(rem) + "day" + output

        # WEEK
        if rest != 0:
            output = "+" + text_type(rest) + "week" + output

        if isNegative:
            output = output.replace("+", "-")

        # MONTH AND YEAR
        if self.month:
            sign = "-" if self.month < 0 else "+"
            month = abs(self.month)

            if month <= 18 and month != 12:
                output = sign + text_type(month) + "month" + output
            else:
                m = month % 12
                if m != 0:
                    output = sign + text_type(m) + "month" + output
                y = floor(month / 12)
                output = sign + text_type(y) + "year" + output

        if output[0] == "+":
            output = output[1::]
        if output[0] == '1' and not is_number(output[1]):
            output = output[1::]
        return output


    def format(self, interval, decimal):
        return self.round(Duration(interval), decimal) + interval

    def round(self, interval, decimal=0):
        output = self / interval
        output = round(output, decimal)
        return output


def _string2Duration(text):
    """
    CONVERT SIMPLE <float><type> TO A DURATION OBJECT
    """
    if text == "" or text == "zero":
        return ZERO

    amount, interval = re.match(r"([\d\.]*)(.*)", text).groups()
    amount = int(amount) if amount else 1

    if MILLI_VALUES[interval] == None:
        from mo_logs import Log
        Log.error(
            "{{interval|quote}} in {{text|quote}} is not a recognized duration type (did you use the pural form by mistake?",
            interval=interval,
            text=text
        )

    output = Duration(0)
    if MONTH_VALUES[interval] == 0:
        output.milli = amount * MILLI_VALUES[interval]
    else:
        output.milli = amount * MONTH_VALUES[interval] * MILLI_VALUES.month
        output.month = amount * MONTH_VALUES[interval]

    return output


def parse(value):
    output = Duration(0)

    # EXPECTING CONCAT OF <sign><integer><type>
    plist = value.split("+")
    for p, pplist in enumerate(plist):
        mlist = pplist.split("-")
        output = output + _string2Duration(mlist[0])
        for m in mlist[1::]:
            output = output - _string2Duration(m)
    return output


MILLI_VALUES = wrap({
    "year": float(52 * 7 * 24 * 60 * 60 * 1000),  # 52weeks
    "quarter": float(13 * 7 * 24 * 60 * 60 * 1000),  # 13weeks
    "month": float(28 * 24 * 60 * 60 * 1000),  # 4weeks
    "week": float(7 * 24 * 60 * 60 * 1000),
    "day": float(24 * 60 * 60 * 1000),
    "hour": float(60 * 60 * 1000),
    "minute": float(60 * 1000),
    "second": float(1000),
    "milli": float(1),
    "zero": float(0)
})

MONTH_VALUES = wrap({
    "year": 12,
    "quarter": 3,
    "month": 1,
    "week": 0,
    "day": 0,
    "hour": 0,
    "minute": 0,
    "second": 0,
    "milli": 0
})

# A REAL MONTH IS LARGER THAN THE CANONICAL MONTH
MONTH_SKEW = MILLI_VALUES["year"] / 12 - MILLI_VALUES.month


def compare(a, b):
    return a.milli - b.milli


DOMAIN = {
    "type": "duration",
    "compare": compare
}

ZERO = Duration(0)
SECOND = Duration("second")
MINUTE = Duration("minute")
HOUR = Duration("hour")
DAY = Duration("day")
WEEK = Duration("week")
MONTH = Duration("month")
QUARTER = Duration("quarter")
YEAR = Duration("year")

COMMON_INTERVALS = [
    Duration("second"),
    Duration("15second"),
    Duration("30second"),
    Duration("minute"),
    Duration("5minute"),
    Duration("15minute"),
    Duration("30minute"),
    Duration("hour"),
    Duration("2hour"),
    Duration("3hour"),
    Duration("6hour"),
    Duration("12hour"),
    Duration("day"),
    Duration("2day"),
    Duration("week"),
    Duration("2week"),
    Duration("month"),
    Duration("2month"),
    Duration("quarter"),
    Duration("6month"),
    Duration("year")
]

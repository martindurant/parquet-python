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

from mo_future import text_type, PY3
from mo_logs import Log
from mo_logs.strings import expand_template


class SQL(text_type):
    """
    ACTUAL SQL, DO NOT QUOTE THIS STRING
    """
    def __init__(self, template='', param=None):
        text_type.__init__(self)
        self.template = template
        self.param = param

    @property
    def sql(self):
        return expand_template(self.template, self.param)

    def __add__(self, other):
        if not isinstance(other, SQL):
            if isinstance(other, text_type) and all(c not in other for c in ('"', '\'', '`')):
               return SQL(self.sql + other)
            Log.error("Can only concat other SQL")
        else:
            return SQL(self.sql+other.sql)

    def __radd__(self, other):
        if not isinstance(other, SQL):
            if isinstance(other, text_type) and all(c not in other for c in ('"', '\'', '`')):
                return SQL(other + self.sql)
            Log.error("Can only concat other SQL")
        else:
            return SQL(other.sql + self.sql)

    def join(self, list):
        if not all(isinstance(s, SQL) for s in list):
            Log.error("Can only join other SQL")
        return SQL(self.sql.join(list))

    if PY3:
        def __bytes__(self):
            Log.error("do not do this")
    else:
        def __str__(self):
            Log.error("do not do this")


class DB(object):

    def quote_column(self, column_name, table=None):
        raise NotImplementedError()

    def db_type_to_json_type(self, type):
        raise NotImplementedError()


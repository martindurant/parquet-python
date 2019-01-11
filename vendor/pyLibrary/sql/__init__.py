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
from mo_future import PY3, text_type
from mo_logs import Log
import pyLibrary.sql


class SQL(text_type):
    """
    ACTUAL SQL, DO NOT QUOTE THIS STRING
    """
    def __init__(self, value):
        text_type.__init__(self)
        if isinstance(value, SQL):
            Log.error("Expecting text, not SQL")
        self.value = value

    @property
    def sql(self):
        return self.value  # expand_template(self.template, self.param)

    def __add__(self, other):
        if not isinstance(other, SQL):
            if is_text(other) and all(c not in other for c in ('"', '\'', '`')):
               return SQL(self.sql + other)
            Log.error("Can only concat other SQL")
        else:
            return SQL(self.sql+other.sql)

    def __radd__(self, other):
        if not isinstance(other, SQL):
            if is_text(other) and all(c not in other for c in ('"', '\'', '`')):
                return SQL(other + self.sql)
            Log.error("Can only concat other SQL")
        else:
            return SQL(other.sql + self.sql)

    def join(self, list_):
        list_ = list(list_)
        if not all(isinstance(s, SQL) for s in list_):
            Log.error("Can only join other SQL")
        return SQL(self.sql.join(list_))

    if PY3:
        def __str__(self):
            return self.sql

        def __bytes__(self):
            Log.error("do not do this")
    else:
        def __unicode__(self):
            return self.sql

        def __str__(self):
            Log.error("do not do this")

    def __data__(self):
        return self.sql


SQL_STAR = SQL(" * ")

SQL_AND = SQL(" AND ")
SQL_OR = SQL(" OR ")
SQL_NOT = SQL(" NOT ")
SQL_ON = SQL(" ON ")

SQL_CASE = SQL(" CASE ")
SQL_WHEN = SQL(" WHEN ")
SQL_THEN = SQL(" THEN ")
SQL_ELSE = SQL(" ELSE ")
SQL_END = SQL(" END ")

SQL_COMMA = SQL(", ")
SQL_UNION_ALL = SQL("\nUNION ALL\n")
SQL_UNION = SQL("\nUNION\n")
SQL_LEFT_JOIN = SQL("\nLEFT JOIN\n")
SQL_INNER_JOIN = SQL("\nJOIN\n")
SQL_EMPTY_STRING = SQL("''")
SQL_TRUE = SQL(" 1 ")
SQL_FALSE = SQL(" 0 ")
SQL_ONE = SQL(" 1 ")
SQL_ZERO = SQL(" 0 ")
SQL_NEG_ONE = SQL(" -1 ")
SQL_NULL = SQL(" NULL ")
SQL_IS_NULL = SQL(" IS NULL ")
SQL_IS_NOT_NULL = SQL(" IS NOT NULL ")
SQL_SELECT = SQL("\nSELECT\n")
SQL_FROM = SQL("\nFROM\n")
SQL_WHERE = SQL("\nWHERE\n")
SQL_GROUPBY = SQL("\nGROUP BY\n")
SQL_ORDERBY = SQL("\nORDER BY\n")
SQL_DESC = SQL(" DESC ")
SQL_ASC = SQL(" ASC ")
SQL_LIMIT = SQL("\nLIMIT\n")


class DB(object):

    def quote_column(self, column_name, table=None):
        raise NotImplementedError()

    def db_type_to_json_type(self, type):
        raise NotImplementedError()

def sql_list(list_):
    list_ = list(list_)
    if not all(isinstance(s, SQL) for s in list_):
        Log.error("Can only join other SQL")
    return SQL(" " + ", ".join(l.value for l in list_) + " ")


def sql_iso(sql):
    return "("+sql+")"


def sql_count(sql):
    return "COUNT(" + sql + ")"


def sql_concat(list_):
    return SQL(" || ").join(sql_iso(l) for l in list_)


def quote_set(list_):
    return sql_iso(sql_list(map(pyLibrary.sql.sqlite.quote_value, list_)))


def sql_alias(value, alias):
    return SQL(value.template + " AS " + alias.template)


def sql_coalesce(list_):
    return "COALESCE(" + SQL_COMMA.join(list_) + ")"


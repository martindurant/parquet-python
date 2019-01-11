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

from copy import copy
from math import isnan

from mo_dots import Data, data_types, listwrap
from mo_dots.lists import list_types
from mo_future import boolean_type, long, none_type, text_type
from mo_logs import Log
from mo_times import Date

builtin_tuple = tuple

Expression = None
expression_module = None
JX = None


_next_id = 0


def next_id():
    global _next_id
    try:
        return _next_id
    finally:
        _next_id+=1


def all_bases(bases):
    for b in bases:
        yield b
        for y in all_bases(b.__bases__):
            yield y


# EVERY OPERATOR WILL HAVE lang WHICH POINTS TO LANGUAGE
class LanguageElement(type):
    def __new__(cls, name, bases, dct):
        x = type.__new__(cls, name, bases, dct)
        x.lang = None
        if x.__module__ == expression_module:
            # ALL OPS IN expression_module ARE GIVEN AN ID, NO OTHERS
            x.id = next_id()
        return x

    def __init__(cls, *args):
        global Expression, expression_module
        type.__init__(cls, *args)
        if not expression_module and cls.__name__ == "Expression":
            # THE expression_module IS DETERMINED BY THE LOCATION OF Expression CLASS
            Expression = cls
            expression_module = cls.__module__


BaseExpression = LanguageElement(str("BaseExpression"), (object,), {})


class Language(object):

    def __init__(self, name):
        self.name = name
        self.ops = None

    def __getitem__(self, item):
        class_ = self.ops[item.id]
        if class_.__name__ != item.__class__.__name__:
            Log.error("programming error")
        item.__class__ = class_
        return item

    def __str__(self):
        return self.name


def define_language(lang_name, module_vars):
    # LET ALL EXPRESSIONS POINT TO lang OBJECT WITH ALL EXPRESSIONS
    # ENSURE THIS IS BELOW ALL SUB_CLASS DEFINITIONS SO var() CAPTURES ALL EXPRESSIONS
    global JX

    if lang_name:
        language = Language(lang_name)
        language.ops = copy(JX.ops)
    else:
        num_ops = 1 + max(
            obj.id
            for obj in module_vars.values() if isinstance(obj, type) and hasattr(obj, 'id')
        )
        language = JX = Language("JX")
        language.ops = [None] * num_ops

    for _, new_op in module_vars.items():
        if isinstance(new_op, type) and hasattr(new_op, 'id'):
            # EXPECT OPERATORS TO HAVE id
            # EXPECT NEW DEFINED OPS IN THIS MODULE TO HAVE lang NOT SET
            curr = getattr(new_op, "lang")
            if not curr:
                old_op = language.ops[new_op.id]
                if old_op is not None and old_op.__name__ != new_op.__name__:
                    Log.error("Logic error")
                language.ops[new_op.id] = new_op
                setattr(new_op, "lang", language)

    if lang_name:
        # ENSURE THE ALL OPS ARE DEFINED ON THE NEW LANGUAGE
        for base_op, new_op in list(zip(JX.ops, language.ops)):
            if new_op is base_op:
                # MISSED DEFINITION, ADD ONE
                new_op = type(base_op.__name__, (base_op,), {})
                language.ops[new_op.id] = new_op
                setattr(new_op, "lang", language)

    return language


def is_op(call, op):
    """
    :param call: The specific operator instance (a method call)
    :param op: The the operator we are testing against
    :return: isinstance(call, op), but faster
    """
    try:
        return call.id == op.id
    except Exception as e:
        return False


def is_expression(call):
    try:
        output = getattr(call, 'id', None) != None
    except Exception:
        output = False
    if output != isinstance(call, Expression):
        Log.error("programmer error")
    return output


def value_compare(left, right, ordering=1):
    """
    SORT VALUES, NULL IS THE LEAST VALUE
    :param left: LHS
    :param right: RHS
    :param ordering: (-1, 0, 1) TO AFFECT SORT ORDER
    :return: The return value is negative if x < y, zero if x == y and strictly positive if x > y.
    """

    try:
        ltype = left.__class__
        rtype = right.__class__

        if ltype in list_types or rtype in list_types:
            if left == None:
                return ordering
            elif right == None:
                return - ordering

            left = listwrap(left)
            right = listwrap(right)
            for a, b in zip(left, right):
                c = value_compare(a, b) * ordering
                if c != 0:
                    return c

            if len(left) < len(right):
                return - ordering
            elif len(left) > len(right):
                return ordering
            else:
                return 0

        if ltype is float and isnan(left):
            left = None
            ltype = none_type
        if rtype is float and isnan(right):
            right = None
            rtype = none_type

        null_order = ordering*10
        ltype_num = TYPE_ORDER.get(ltype, null_order)
        rtype_num = TYPE_ORDER.get(rtype, null_order)

        type_diff = ltype_num - rtype_num
        if type_diff != 0:
            return ordering if type_diff > 0 else -ordering

        if ltype_num == null_order:
            return 0
        elif ltype is builtin_tuple:
            for a, b in zip(left, right):
                c = value_compare(a, b)
                if c != 0:
                    return c * ordering
            return 0
        elif ltype in data_types:
            for k in sorted(set(left.keys()) | set(right.keys())):
                c = value_compare(left.get(k), right.get(k)) * ordering
                if c != 0:
                    return c
            return 0
        elif left > right:
            return ordering
        elif left < right:
            return -ordering
        else:
            return 0
    except Exception as e:
        Log.error("Can not compare values {{left}} to {{right}}", left=left, right=right, cause=e)


TYPE_ORDER = {
    boolean_type: 0,
    int: 1,
    float: 1,
    Date: 1,
    long: 1,
    text_type: 2,
    list: 3,
    builtin_tuple: 3,
    dict: 4,
    Data: 4
}




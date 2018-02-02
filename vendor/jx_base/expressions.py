# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import itertools
import operator
from collections import Mapping
from decimal import Decimal

import mo_json
from jx_base import OBJECT, python_type_to_json_type, BOOLEAN, NUMBER, INTEGER, STRING
from jx_base.queries import is_variable_name, get_property_name
from mo_dots import coalesce, wrap, Null, split_field
from mo_future import text_type, utf8_json_encoder, get_function_name
from mo_json import scrub
from mo_logs import Log, Except
from mo_math import Math, MAX, MIN, UNION
from mo_times.dates import Date, unicode2Date

ALLOW_SCRIPTING = False
EMPTY_DICT = {}


def extend(cls):
    """
    DECORATOR TO ADD METHODS TO CLASSES
    :param cls: THE CLASS TO ADD THE METHOD TO
    :return:
    """
    def extender(func):
        setattr(cls, get_function_name(func), func)
        return func
    return extender


def simplified(func):
    def mark_as_simple(self):
        if self.simplified:
            return self

        output = func(self)
        output.simplified = True
        return output
    return mark_as_simple


def jx_expression(expr):
    """
    WRAP A JSON EXPRESSION WITH OBJECT REPRESENTATION
    """
    if isinstance(expr, Expression):
        Log.error("Expecting JSON, not expression")

    if expr in (True, False, None) or expr == None or isinstance(expr, (float, int, Decimal, Date)):
        return Literal(None, expr)
    elif isinstance(expr, text_type):
        return Variable(expr)
    elif isinstance(expr, (list, tuple)):
        return TupleOp("tuple", map(jx_expression, expr))  # FORMALIZE

    expr = wrap(expr)
    try:
        items = expr.items()
    except Exception as e:
        Log.error("programmer error expr = {{value|quote}}", value=expr, cause=e)

    for item in items:
        op, term = item
        class_ = operators.get(op)
        if class_:
            return class_.define(expr)
    else:
        if not items:
            return NULL
        raise Log.error("{{operator|quote}} is not a known operator", operator=op)


class Expression(object):
    data_type = OBJECT
    has_simple_form = False

    def __init__(self, op, terms):
        self.simplified = False
        if isinstance(terms, (list, tuple)):
            if not all(isinstance(t, Expression) for t in terms):
                Log.error("Expecting an expression")
        elif isinstance(terms, Mapping):
            if not all(isinstance(k, Variable) and isinstance(v, Literal) for k, v in terms.items()):
                Log.error("Expecting an {<variable>: <literal>}")
        elif terms == None:
            pass
        else:
            if not isinstance(terms, Expression):
                Log.error("Expecting an expression")

    @classmethod
    def define(cls, expr):
        """
        GENERAL SUPPORT FOR BUILDING EXPRESSIONS FROM JSON EXPRESSIONS
        OVERRIDE THIS IF AN OPERATOR EXPECTS COMPLICATED PARAMETERS
        :param expr: Data representing a JSON Expression
        :return: Python parse tree
        """

        try:
            items = expr.items()
        except Exception as e:
            Log.error("programmer error expr = {{value|quote}}", value=expr, cause=e)

        for item in items:
            op, term = item
            class_ = operators.get(op)
            if class_:
                clauses = {k: jx_expression(v) for k, v in expr.items() if k != op}
                break
        else:
            if not items:
                return NULL
            raise Log.error("{{operator|quote}} is not a known operator", operator=op)

        if term == None:
            return class_(op, [], **clauses)
        elif isinstance(term, list):
            terms = list(map(jx_expression, term))
            return class_(op, terms, **clauses)
        elif isinstance(term, Mapping):
            items = term.items()
            if class_.has_simple_form:
                if len(items) == 1:
                    k, v = items[0]
                    return class_(op, [Variable(k), Literal(None, v)], **clauses)
                else:
                    return class_(op, {k: Literal(None, v) for k, v in items}, **clauses)
            else:
                return class_(op, jx_expression(term), **clauses)
        else:
            if op in ["literal", "date", "offset"]:
                return class_(op, term, **clauses)
            else:
                return class_(op, jx_expression(term), **clauses)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def many(self):
        """
        :return: True IF THE EXPRESSION RETURNS A MULTIVALUE (WHICH IS NOT A LIST OR A TUPLE)
        """
        return False

    def __data__(self):
        raise NotImplementedError

    def vars(self):
        raise Log.error("{{type}} has no `vars` method", type=self.__class__.__name__)

    def map(self, map):
        raise Log.error("{{type}} has no `map` method", type=self.__class__.__name__)

    def missing(self):
        """
        THERE IS PLENTY OF OPPORTUNITY TO SIMPLIFY missing EXPRESSIONS
        OVERRIDE THIS METHOD TO SIMPLIFY
        :return:
        """
        if self.type == BOOLEAN:
            Log.error("programmer error")
        return MissingOp("missing", self)

    def exists(self):
        """
        THERE IS PLENTY OF OPPORTUNITY TO SIMPLIFY exists EXPRESSIONS
        OVERRIDE THIS METHOD TO SIMPLIFY
        :return:
        """
        return NotOp("not", self.missing())

    def is_true(self):
        """
        :return: True, IF THIS EXPRESSION ALWAYS RETURNS BOOLEAN true
        """
        return FALSE  # GOOD DEFAULT ASSUMPTION

    def is_false(self):
        """
        :return: True, IF THIS EXPRESSION ALWAYS RETURNS BOOLEAN false
        """
        return FALSE  # GOOD DEFAULT ASSUMPTION

    def partial_eval(self):
        """
        ATTEMPT TO SIMPLIFY THE EXPRESSION:
        PREFERABLY RETURNING A LITERAL, BUT MAYBE A SIMPLER EXPRESSION, OR self IF NOT POSSIBLE
        """
        self.simplified = True
        return self

    @property
    def type(self):
        return self.data_type

    def __eq__(self, other):
        Log.note("this is slow on {{type}}", type=text_type(self.__class__.__name__))
        if other is None:
            return False
        return self.__data__() == other.__data__()


class Variable(Expression):

    def __init__(self, var):
        """
        :param var:  DOT DELIMITED PATH INTO A DOCUMENT
        :param verify: True - VERIFY THIS IS A VALID NAME (use False for trusted code only)
        """
        Expression.__init__(self, "", None)
        self.var = get_property_name(var)

    def __call__(self, row, rownum=None, rows=None):
        path = split_field(self.var)
        for p in path:
            row = row.get(p)
            if row is None:
                return None
        if isinstance(row, list) and len(row) == 1:
            return row[0]
        return row

    def __data__(self):
        return self.var

    @property
    def many(self):
        return True

    def vars(self):
        return {self.var}

    def map(self, map_):
        if not isinstance(map_, Mapping):
            Log.error("Expecting Mapping")

        return Variable(coalesce(map_.get(self.var), self.var))

    def __hash__(self):
        return self.var.__hash__()

    def __eq__(self, other):
        return self.var.__eq__(other)

    def __unicode__(self):
        return self.var

    def __str__(self):
        return str(self.var)
IDENTITY = Variable(".")


class OffsetOp(Expression):
    """
    OFFSET INDEX INTO A TUPLE
    """

    def __init__(self, op, var):
        Expression.__init__(self, "offset", None)
        if not Math.is_integer(var):
            Log.error("Expecting an integer")
        self.var = var

    def __call__(self, row, rownum=None, rows=None):
        try:
            return row[self.var]
        except Exception:
            return None

    def __data__(self):
        return {"offset": self.var}

    def vars(self):
        return {}

    def __hash__(self):
        return self.var.__hash__()

    def __eq__(self, other):
        return self.var == other

    def __unicode__(self):
        return text_type(self.var)

    def __str__(self):
        return str(self.var)


class RowsOp(Expression):
    has_simple_form = True

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.var, self.offset = term
        if isinstance(self.var, Variable):
            if isinstance(self.var, Variable) and not any(self.var.var.startswith(p) for p in ["row.", "rows.", "rownum"]):  # VARIABLES ARE INTERPRETED LITERALLY
                self.var = Literal("literal", self.var.var)
            else:
                Log.error("can not handle")
        else:
            Log.error("can not handle")

    def __data__(self):
        if isinstance(self.var, Literal) and isinstance(self.offset, Literal):
            return {"rows": {self.var.json, self.offset.value}}
        else:
            return {"rows": [self.var.__data__(), self.offset.__data__()]}

    def vars(self):
        return self.var.vars() | self.offset.vars() | {"rows", "rownum"}

    def map(self, map_):
        return BinaryOp("rows", [self.var.map(map_), self.offset.map(map_)])


class GetOp(Expression):
    has_simple_form = True

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.var, self.offset = term

    def __data__(self):
        if isinstance(self.var, Literal) and isinstance(self.offset, Literal):
            return {"get": {self.var.json, self.offset.value}}
        else:
            return {"get": [self.var.__data__(), self.offset.__data__()]}

    def vars(self):
        return self.var.vars() | self.offset.vars()

    def map(self, map_):
        return BinaryOp("get", [self.var.map(map_), self.offset.map(map_)])


class SelectOp(Expression):
    has_simple_form = True

    def __init__(self, op, terms):
        self.terms = terms

    @classmethod
    def define(cls, expr):
        term = expr.select
        terms = []
        if not isinstance(term, list):
            raise Log.error("Expecting a list")
        for t in term:
            if isinstance(t, text_type):
                if not is_variable_name(t):
                    Log.error("expecting {{value}} a simple dot-delimited path name", value=t)
                terms.append({"name": t, "value": jx_expression(t)})
            elif t.name == None:
                if t.value == None:
                    Log.error("expecting select parameters to have name and value properties")
                elif isinstance(t.value, text_type):
                    if not is_variable_name(t):
                        Log.error("expecting {{value}} a simple dot-delimited path name", value=t.value)
                    else:
                        terms.append({"name": t.value, "value": jx_expression(t.value)})
                else:
                    Log.error("expecting a name property")
            else:
                terms.append({"name": t.name, "value": jx_expression(t.value)})
        return SelectOp("select", terms)

    def __data__(self):
        return {"select": [
            {
                "name": t.name.__data__(),
                "value": t.value.__data__()
            }
            for t in self.terms
        ]}

    def vars(self):
        return UNION(t.value for t in self.terms)

    def map(self, map_):
        return SelectOp("select", [
            {"name": t.name, "value": t.value.map(map_)}
            for t in self.terms
        ])


class ScriptOp(Expression):
    """
    ONLY FOR WHEN YOU TRUST THE SCRIPT SOURCE
    """

    def __init__(self, op, script):
        Expression.__init__(self, op, None)
        if not isinstance(script, text_type):
            Log.error("expecting text of a script")
        self.simplified = True
        self.script = script

    @classmethod
    def define(cls, expr):
        if ALLOW_SCRIPTING:
            Log.warning("Scripting has been activated:  This has known security holes!!\nscript = {{script|quote}}", script=expr.script.term)
            return ScriptOp("script", expr.script)
        else:
            Log.error("scripting is disabled")

    def vars(self):
        return set()

    def map(self, map_):
        return self

    def __unicode__(self):
        return self.script

    def __str__(self):
        return str(self.script)


_json_encoder = utf8_json_encoder


def value2json(value):
    try:
        scrubbed = scrub(value, scrub_number=float)
        return text_type(_json_encoder(scrubbed))
    except Exception as e:
        e = Except.wrap(e)
        Log.warning("problem serializing {{type}}", type=text_type(repr(value)), cause=e)
        raise e


class Literal(Expression):
    """
    A literal JSON document
    """

    def __new__(cls, op, term):
        if term == None:
            return NULL
        if term is True:
            return TRUE
        if term is False:
            return FALSE
        if isinstance(term, Mapping) and term.date:
            # SPECIAL CASE
            return DateOp(None, term.date)
        return object.__new__(cls)

    def __init__(self, op, term):
        Expression.__init__(self, "", None)
        self.simplified = True
        self.term = term

    @classmethod
    def define(cls, expr):
        return Literal(None, expr.literal)

    def __nonzero__(self):
        return True

    def __eq__(self, other):
        if other == None:
            if self.term == None:
                return True
            else:
                return False
        elif self.term == None:
            return False

        Log.warning("expensive")

        from mo_testing.fuzzytestcase import assertAlmostEqual

        try:
            assertAlmostEqual(self.term, other)
            return True
        except Exception:
            return False

    def __data__(self):
        return {"literal": self.value}

    @property
    def value(self):
        return self.term

    @property
    def json(self):
        if self.term == "":
            self._json = '""'
        else:
            self._json = value2json(self.term)

        return self._json

    def vars(self):
        return set()

    def map(self, map_):
        return self

    def missing(self):
        if self.term in [None, Null]:
            return TRUE
        if self.value == '':
            return TRUE
        return FALSE

    def __call__(self, row=None, rownum=None, rows=None):
        return self.value

    def __unicode__(self):
        return self._json

    def __str__(self):
        return str(self._json)

    @property
    def type(self):
        return python_type_to_json_type[self.term.__class__]

    def partial_eval(self):
        return self
ZERO = Literal("literal", 0)


class NullOp(Literal):
    """
    FOR USE WHEN EVERYTHING IS EXPECTED TO BE AN Expression
    USE IT TO EXPECT A NULL VALUE IN assertAlmostEqual
    """
    data_type = OBJECT

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, op=None, term=None):
        Literal.__init__(self, op, None)

    def __nonzero__(self):
        return False

    def __eq__(self, other):
        return other == None

    def __gt__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __ge__(self, other):
        if other == None:
            return True
        return False

    def __le__(self, other):
        if other == None:
            return True
        return False

    def __data__(self):
        return {"null": {}}

    def vars(self):
        return set()

    def map(self, map_):
        return self

    def missing(self):
        return TRUE

    def exists(self):
        return FALSE

    def __call__(self, row=None, rownum=None, rows=None):
        return Null

    def __unicode__(self):
        return "null"

    def __str__(self):
        return b"null"

    def __data__(self):
        return None
NULL = NullOp()


class TrueOp(Literal):
    data_type = BOOLEAN

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, op=None, term=None):
        Literal.__init__(self, op, True)

    @classmethod
    def define(cls, expr):
        return TRUE

    def __nonzero__(self):
        return True

    def __eq__(self, other):
        return (other is TRUE) or (other is True)

    def __data__(self):
        return True

    def vars(self):
        return set()

    def map(self, map_):
        return self

    def missing(self):
        return FALSE

    def is_true(self):
        return TRUE

    def is_false(self):
        return FALSE

    def __call__(self, row=None, rownum=None, rows=None):
        return True

    def __unicode__(self):
        return "true"

    def __str__(self):
        return b"true"
TRUE = TrueOp()


class FalseOp(Literal):
    data_type = BOOLEAN

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, op=None, term=None):
        Literal.__init__(self, op, False)

    @classmethod
    def define(cls, expr):
        return FALSE

    def __nonzero__(self):
        return False

    def __eq__(self, other):
        return (other is FALSE) or (other is False)

    def __data__(self):
        return False

    def vars(self):
        return set()

    def map(self, map_):
        return self

    def missing(self):
        return FALSE

    def is_true(self):
        return FALSE

    def is_false(self):
        return TRUE

    def __call__(self, row=None, rownum=None, rows=None):
        return False

    def __unicode__(self):
        return "false"

    def __str__(self):
        return b"false"
FALSE = FalseOp()


class DateOp(Literal):
    date_type = NUMBER

    def __init__(self, op, term):
        if hasattr(self, "date"):
            return
        self.date = term
        v = unicode2Date(self.date)
        if isinstance(v, Date):
            Literal.__init__(self, op, v.unix)
        else:
            Literal.__init__(self, op, v.seconds)

    @classmethod
    def define(cls, expr):
        return DateOp("date", expr.date)

    def __data__(self):
        return {"date": self.date}

    def __call__(self, row=None, rownum=None, rows=None):
        return Date(self.date)


class TupleOp(Expression):
    date_type = OBJECT

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        if terms == None:
            self.terms = []
        elif isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

    def __data__(self):
        return {"tuple": [t.__data__() for t in self.terms]}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return TupleOp("tuple", [t.map(map_) for t in self.terms])

    def missing(self):
        return FALSE


class LeavesOp(Expression):
    date_type = OBJECT

    def __init__(self, op, term, prefix=None):
        Expression.__init__(self, op, term)
        self.term = term
        self.prefix = prefix

    def __data__(self):
        if self.prefix:
            return {"leaves": self.term.__data__(), "prefix": self.prefix}
        else:
            return {"leaves": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return LeavesOp("leaves", self.term.map(map_))

    def missing(self):
        return FALSE


class BinaryOp(Expression):
    has_simple_form = True
    data_type = NUMBER

    operators = {
        "sub": "-",
        "subtract": "-",
        "minus": "-",
        "mul": "*",
        "mult": "*",
        "multiply": "*",
        "div": "/",
        "divide": "/",
        "exp": "**",
        "mod": "%"
    }

    def __init__(self, op, terms, default=NULL):
        Expression.__init__(self, op, terms)
        if op not in BinaryOp.operators:
            Log.error("{{op|quote}} not a recognized operator", op=op)
        self.op = op
        self.lhs, self.rhs = terms
        self.default = default

    @property
    def name(self):
        return self.op;

    def __data__(self):
        if isinstance(self.lhs, Variable) and isinstance(self.rhs, Literal):
            return {self.op: {self.lhs.var, self.rhs.value}, "default": self.default}
        else:
            return {self.op: [self.lhs.__data__(), self.rhs.__data__()], "default": self.default}

    def vars(self):
        return self.lhs.vars() | self.rhs.vars() | self.default.vars()

    def map(self, map_):
        return BinaryOp(self.op, [self.lhs.map(map_), self.rhs.map(map_)], default=self.default.map(map_))

    def missing(self):
        if self.default.exists():
            return FALSE
        else:
            return OrOp("or", [self.lhs.missing(), self.rhs.missing()])

    # @simplified
    # def partial_eval(self):
    #     lhs = FirstOp("first", self.lhs).partial_eval()
    #     rhs = FirstOp("first", self.rhs).partial_eval()
    #     default_ = FirstOp("first", self.default).partial_eval()
    #     return BinaryOp(self.op, [lhs, rhs], default=default_)


class InequalityOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    operators = {
        "gt": ">",
        "gte": ">=",
        "lte": "<=",
        "lt": "<"
    }

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        if op not in InequalityOp.operators:
            Log.error("{{op|quote}} not a recognized operator", op=op)
        self.op = op
        self.lhs, self.rhs = terms

    @property
    def name(self):
        return self.op;

    def __data__(self):
        if isinstance(self.lhs, Variable) and isinstance(self.rhs, Literal):
            return {self.op: {self.lhs.var, self.rhs.value}}
        else:
            return {self.op: [self.lhs.__data__(), self.rhs.__data__()]}

    def vars(self):
        return self.lhs.vars() | self.rhs.vars()

    def map(self, map_):
        return InequalityOp(self.op, [self.lhs.map(map_), self.rhs.map(map_)])

    def missing(self):
        return FALSE

    @simplified
    def partial_eval(self):
        lhs = self.lhs.partial_eval()
        rhs = self.rhs.partial_eval()

        if isinstance(lhs, Literal) and isinstance(rhs, Literal):
            return Literal(None, builtin_ops[self.op](lhs, rhs))

        return InequalityOp(self.op, [lhs, rhs])


class DivOp(Expression):
    has_simple_form = True
    data_type = NUMBER

    def __init__(self, op, terms, default=NULL):
        Expression.__init__(self, op, terms)
        self.lhs, self.rhs = terms
        self.default = default

    def __data__(self):
        if isinstance(self.lhs, Variable) and isinstance(self.rhs, Literal):
            return {"div": {self.lhs.var, self.rhs.value}, "default": self.default}
        else:
            return {"div": [self.lhs.__data__(), self.rhs.__data__()], "default": self.default}

    def vars(self):
        return self.lhs.vars() | self.rhs.vars() | self.default.vars()

    def map(self, map_):
        return DivOp("div", [self.lhs.map(map_), self.rhs.map(map_)], default=self.default.map(map_))

    def missing(self):
        AndOp("and", [
            self.default.missing(),
            OrOp("or", [self.lhs.missing(), self.rhs.missing(), EqOp("eq", [self.rhs, ZERO])])
        ]).partial_eval()


class FloorOp(Expression):
    has_simple_form = True
    data_type = NUMBER

    def __init__(self, op, terms, default=NULL):
        Expression.__init__(self, op, terms)
        self.lhs, self.rhs = terms
        self.default = default

    def __data__(self):
        if isinstance(self.lhs, Variable) and isinstance(self.rhs, Literal):
            return {"floor": {self.lhs.var, self.rhs.value}, "default": self.default}
        else:
            return {"floor": [self.lhs.__data__(), self.rhs.__data__()], "default": self.default}

    def vars(self):
        return self.lhs.vars() | self.rhs.vars() | self.default.vars()

    def map(self, map_):
        return FloorOp("floor", [self.lhs.map(map_), self.rhs.map(map_)], default=self.default.map(map_))

    def missing(self):
        if self.default.exists():
            return FALSE
        else:
            return OrOp("or", [self.lhs.missing(), self.rhs.missing(), EqOp("eq", [self.rhs, ZERO])])


class EqOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    def __new__(cls, op, terms):
        if isinstance(terms, list):
            return object.__new__(cls)

        items = terms.items()
        if len(items) == 1:
            if isinstance(items[0][1], list):
                return InOp("in", items[0])
            else:
                return EqOp("eq", items[0])
        else:
            acc = []
            for lhs, rhs in items:
                if rhs.json.startswith("["):
                    acc.append(InOp("in", [Variable(lhs), rhs]))
                else:
                    acc.append(EqOp("eq", [Variable(lhs), rhs]))
            return AndOp("and", acc)

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        self.op = op
        self.lhs, self.rhs = terms

    def __data__(self):
        if isinstance(self.lhs, Variable) and isinstance(self.rhs, Literal):
            return {"eq": {self.lhs.var, self.rhs.value}}
        else:
            return {"eq": [self.lhs.__data__(), self.rhs.__data__()]}

    def vars(self):
        return self.lhs.vars() | self.rhs.vars()

    def map(self, map_):
        return EqOp(self.op, [self.lhs.map(map_), self.rhs.map(map_)])

    def missing(self):
        return FALSE

    def exists(self):
        return TRUE

    @simplified
    def partial_eval(self):
        lhs = self.lhs.partial_eval()
        rhs = self.rhs.partial_eval()

        if isinstance(lhs, Literal) and isinstance(rhs, Literal):
            return TRUE if builtin_ops["eq"](lhs.value, rhs.value) else FALSE
        else:
            return WhenOp(
                "when",
                lhs.missing(),
                **{
                    "then": rhs.missing(),
                    "else": BasicEqOp("eq", [lhs, rhs])
                }
            ).partial_eval()


class NeOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        if isinstance(terms, (list, tuple)):
            self.lhs, self.rhs = terms
        elif isinstance(terms, Mapping):
            self.rhs, self.lhs = terms.items()[0]
        else:
            Log.error("logic error")

    def __data__(self):
        if isinstance(self.lhs, Variable) and isinstance(self.rhs, Literal):
            return {"ne": {self.lhs.var, self.rhs.value}}
        else:
            return {"ne": [self.lhs.__data__(), self.rhs.__data__()]}

    def vars(self):
        return self.lhs.vars() | self.rhs.vars()

    def map(self, map_):
        return NeOp("ne", [self.lhs.map(map_), self.rhs.map(map_)])

    def missing(self):
        return FALSE  # USING THE decisive EQUAILTY https://github.com/mozilla/jx-sqlite/blob/master/docs/Logical%20Equality.md#definitions

    @simplified
    def partial_eval(self):
        lhs = self.lhs.partial_eval()
        rhs = self.rhs.partial_eval()

        if isinstance(lhs, Literal) and isinstance(rhs, Literal):
            return Literal(None, builtin_ops["ne"](lhs, rhs))

        output = NeOp("ne", [lhs, rhs])
        return output


class NotOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.term = term

    def __data__(self):
        return {"not": self.term.__data__()}

    def __eq__(self, other):
        if not isinstance(other, NotOp):
            return False
        return self.term == other.term

    def vars(self):
            return self.term.vars()

    def map(self, map_):
        return NotOp("not", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        def inverse(term):
            if term is TRUE:
                return FALSE
            elif term is FALSE:
                return TRUE
            elif isinstance(term, NullOp):
                return TRUE
            elif isinstance(term, Literal):
                Log.error("`not` operator expects a Boolean term")
            elif isinstance(term, WhenOp):
                output = WhenOp(
                    "when",
                    term.when,
                    **{"then": inverse(term.then), "else": inverse(term.els_)}
                ).partial_eval()
            elif isinstance(term, AndOp):
                output = OrOp("or", [inverse(t) for t in term.terms]).partial_eval()
            elif isinstance(term, OrOp):
                output = AndOp("and", [inverse(t) for t in term.terms]).partial_eval()
            elif isinstance(term, MissingOp):
                output = NotOp("not", term.expr.missing())
            elif isinstance(term, ExistsOp):
                output = term.field.missing().partial_eval()
            elif isinstance(term, NotOp):
                output = term.term.partial_eval()
            elif isinstance(term, NeOp):
                output = EqOp("eq", [term.lhs, term.rhs]).partial_eval()
            elif isinstance(term, EqOp):
                output = NeOp("ne", [term.lhs, term.rhs]).partial_eval()
            elif isinstance(term, (BasicIndexOfOp, BasicSubstringOp)):
                return FALSE
            else:
                output = NotOp("not", term)

            return output

        output = inverse(self.term.partial_eval())
        return output


class AndOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        if terms == None:
            self.terms = []
        elif isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

    def __data__(self):
        return {"and": [t.__data__() for t in self.terms]}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return AndOp("and", [t.map(map_) for t in self.terms])

    def missing(self):
        return FALSE

    @simplified
    def partial_eval(self):
        terms = []
        ors = []
        for t in self.terms:
            simple = BooleanOp("boolean", t).partial_eval()
            if simple is TRUE:
                pass
            elif simple is FALSE:
                return FALSE
            elif isinstance(simple, AndOp):
                terms.extend([tt for tt in simple.terms if tt not in terms])
            elif isinstance(simple, OrOp):
                ors.append(simple.terms)
            elif simple.type != BOOLEAN:
                Log.error("expecting boolean value")
            elif NotOp("not", simple).partial_eval() in terms:
                return FALSE
            elif simple not in terms:
                terms.append(simple)
        if len(ors) == 0:
            if len(terms) == 0:
                return TRUE
            if len(terms) == 1:
                return terms[0]
            output = AndOp("and", terms)
            return output
        elif len(ors) == 1:  # SOME SIMPLE COMMON FACTORING
            if len(terms) == 0:
                return OrOp("or", ors[0])
            elif len(terms) == 1 and terms[0] in ors[0]:
                return terms[0]
            else:
                agg_terms = []
                for combo in ors[0]:
                    agg_terms.append(
                        AndOp("and", [combo]+terms).partial_eval()
                    )
                return OrOp("or", agg_terms).partial_eval()
        elif len(terms) == 0:
            return OrOp("or", ors[0])

        agg_terms = []
        for combo in itertools.product(*ors):
            agg_terms.append(
                AndOp("and", list(combo)+terms).partial_eval()
            )
        return OrOp("or", agg_terms)


class OrOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        self.terms = terms

    def __data__(self):
        return {"or": [t.__data__() for t in self.terms]}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return OrOp("or", [t.map(map_) for t in self.terms])

    def missing(self):
        return FALSE

    def __call__(self, row=None, rownum=None, rows=None):
        return any(t(row, rownum, rows) for t in self.terms)

    @simplified
    def partial_eval(self):
        terms = []
        ands = []
        for t in self.terms:
            simple = t.partial_eval()
            if simple is TRUE:
                return TRUE
            elif simple in (FALSE, NULL):
                pass
            elif isinstance(simple, OrOp):
                terms.extend(tt for tt in simple.terms if tt not in terms)
            elif isinstance(simple, AndOp):
                ands.append(simple)
            elif simple.type != BOOLEAN:
                Log.error("expecting boolean value")
            elif simple not in terms:
                terms.append(simple)

        if ands:  # REMOVE TERMS THAT ARE MORE RESTRICTIVE THAN OTHERS
            for a in ands:
                for tt in a.terms:
                    if tt in terms:
                        break
                else:
                    terms.append(a)

        if len(terms) == 0:
            return FALSE
        if len(terms) == 1:
            return terms[0]
        return OrOp("or", terms)


class LengthOp(Expression):
    data_type = INTEGER

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"length": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return LengthOp("length", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        term = self.term.partial_eval()
        if isinstance(term, Literal):
            if isinstance(term.value, text_type):
                return Literal(None, len(term.value))
            else:
                return NULL
        else:
            return LengthOp("length", term)

class FirstOp(Expression):
    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term
        self.data_type = self.term.type

    def __data__(self):
        return {"boolean": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return BooleanOp("boolean", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        term = self.term.partial_eval()
        if isinstance(self.term, FirstOp):
            return term
        elif term.type != OBJECT and not term.many:
            return term
        elif term is NULL:
            return term
        elif isinstance(term, Literal):
            Log.error("not handled yet")
        else:
            return FirstOp("first", term)


class BooleanOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"boolean": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return BooleanOp("boolean", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        term = self.term.partial_eval()
        if term is TRUE:
            return TRUE
        elif term in (FALSE, NULL):
            return FALSE
        elif term.type == BOOLEAN:
            return term

        is_missing = term.missing().partial_eval()
        if is_missing is TRUE:
            return FALSE
        elif is_missing is FALSE:
            if term.type in [INTEGER, NUMBER, STRING]:
                return TRUE
        else:
            return NotOp("not", is_missing).partial_eval()


class IsBooleanOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"is_boolean": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return IsBooleanOp("is_boolean", self.term.map(map_))

    def missing(self):
        return FALSE


class IntegerOp(Expression):
    data_type = INTEGER

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"integer": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return IntegerOp("integer", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        term = FirstOp("first", self.term).partial_eval()
        if isinstance(term, CoalesceOp):
            return CoalesceOp("coalesce", [IntegerOp("integer", t) for t in term.terms])
        if term.type == INTEGER:
            return term
        return IntegerOp("integer", term)


class IsIntegerOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"is_integer": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return IsIntegerOp("is_integer", self.term.map(map_))

    def missing(self):
        return FALSE


class NumberOp(Expression):
    data_type = NUMBER

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"number": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return NumberOp("number", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        term = FirstOp("first", self.term).partial_eval()
        if isinstance(term, CoalesceOp):
            return CoalesceOp("coalesce", [NumberOp("number", t) for t in term.terms])
        return self

class IsNumberOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"is_number": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return IsNumberOp("is_number", self.term.map(map_))

    def missing(self):
        return FALSE

    @simplified
    def partial_eval(self):
        term = self.term.partial_eval()

        if isinstance(term, NullOp):
            return FALSE
        elif term.type in (INTEGER, NUMBER):
            return TRUE
        elif term.type == OBJECT:
            return self
        else:
            return FALSE



class StringOp(Expression):
    data_type = STRING

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"string": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return StringOp("string", self.term.map(map_))

    def missing(self):
        return self.term.missing()

    @simplified
    def partial_eval(self):
        term = FirstOp("first", self.term).partial_eval()
        if isinstance(term, CoalesceOp):
            return CoalesceOp("coalesce", [StringOp("string", t).partial_eval() for t in term.terms])
        elif isinstance(term, Literal):
            if term.type == STRING:
                return term
            else:
                return Literal("literal", mo_json.value2json(term.value))
        return self


class IsStringOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.term = term

    def __data__(self):
        return {"is_string": self.term.__data__()}

    def vars(self):
        return self.term.vars()

    def map(self, map_):
        return IsStringOp("is_number", self.term.map(map_))

    def missing(self):
        return FALSE


class CountOp(Expression):
    has_simple_form = False
    data_type = INTEGER

    def __init__(self, op, terms, **clauses):
        Expression.__init__(self, op, terms)
        self.terms = terms

    def __data__(self):
        return {"count": [t.__data__() for t in self.terms]}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return CountOp("count", [t.map(map_) for t in self.terms])

    def missing(self):
        return FALSE

    def exists(self):
        return TrueOp


class MaxOp(Expression):
    data_type = NUMBER

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        if terms == None:
            self.terms = []
        elif isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

    def __data__(self):
        return {"max": [t.__data__() for t in self.terms]}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return MaxOp("max", [t.map(map_) for t in self.terms])

    def missing(self):
        return FALSE

    @simplified
    def partial_eval(self):
        maximum = None
        terms = []
        for t in self.terms:
            simple = t.partial_eval()
            if isinstance(simple, NullOp):
                pass
            elif isinstance(simple, Literal):
                maximum = MAX([maximum, simple.value])
            else:
                terms.append(simple)
        if len(terms) == 0:
            if maximum == None:
                return NULL
            else:
                return Literal(None, maximum)
        else:
            if maximum == None:
                output = MaxOp("max", terms)
            else:
                output = MaxOp("max", [Literal(None, maximum)] + terms)

        return output


class MinOp(Expression):
    data_type = NUMBER

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        if terms == None:
            self.terms = []
        elif isinstance(terms, list):
            self.terms = terms
        else:
            self.terms = [terms]

    def __data__(self):
        return {"min": [t.__data__() for t in self.terms]}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return MinOp("min", [t.map(map_) for t in self.terms])

    def missing(self):
        return FALSE

    @simplified
    def partial_eval(self):
        minimum = None
        terms = []
        for t in self.terms:
            simple = t.partial_eval()
            if isinstance(simple, NullOp):
                pass
            elif isinstance(simple, Literal):
                minimum = MIN([minimum, simple.value])
            else:
                terms.append(simple)
        if len(terms) == 0:
            if minimum == None:
                return NULL
            else:
                return Literal(None, minimum)
        else:
            if minimum == None:
                output = MinOp("min", terms)
            else:
                output = MinOp("min", [Literal(None, minimum)] + terms)

        return output


class MultiOp(Expression):
    has_simple_form = True
    data_type = NUMBER

    operators = {
        "add": (" + ", "0"),  # (operator, zero-array default value) PAIR
        "sum": (" + ", "0"),
        "mul": (" * ", "1"),
        "mult": (" * ", "1"),
        "multiply": (" * ", "1")
    }

    def __init__(self, op, terms, **clauses):
        Expression.__init__(self, op, terms)
        self.op = op
        self.terms = terms
        self.default = coalesce(clauses.get("default"), NULL)
        self.nulls = coalesce(clauses.get("nulls"), FALSE)

    def __data__(self):
        return {self.op: [t.__data__() for t in self.terms], "default": self.default, "nulls": self.nulls}

    def vars(self):
        output = set()
        for t in self.terms:
            output |= t.vars()
        return output

    def map(self, map_):
        return MultiOp(self.op, [t.map(map_) for t in self.terms], **{"default": self.default, "nulls": self.nulls})

    def missing(self):
        if self.nulls:
            if isinstance(self.default, NullOp):
                return AndOp("and", [t.missing() for t in self.terms])
            else:
                return TRUE
        else:
            if isinstance(self.default, NullOp):
                return OrOp("or", [t.missing() for t in self.terms])
            else:
                return FALSE

    def exists(self):
        if self.nulls:
            return OrOp("or", [t.exists() for t in self.terms])
        else:
            return AndOp("and", [t.exists() for t in self.terms])

    @simplified
    def partial_eval(self):
        acc = None
        terms = []
        for t in self.terms:
            simple = t.partial_eval()
            if isinstance(simple, NullOp):
                pass
            elif isinstance(simple, Literal):
                if acc is None:
                    acc = simple.value
                else:
                    acc = builtin_ops[self.op](acc, simple.value)
            else:
                terms.append(simple)
        if len(terms) == 0:
            if acc == None:
                return NULL
            else:
                return Literal(None, acc)
        else:
            if acc is None:
                output = MultiOp(self.op, terms, default=self.default, nulls=self.nulls)
            else:
                output = MultiOp(self.op, [Literal(None, acc)] + terms, default=self.default, nulls=self.nulls)

        return output


def AddOp(op, terms, **clauses):
    return MultiOp("add", terms, **clauses)


def MultOp(op, terms, **clauses):
    return MultiOp("mult", terms, **clauses)


# def MaxOp(op, terms, **clauses):
#     return MaxOp("max", terms, **clauses)
#
#
# def MinOp(op, terms, **clauses):
#     return MinOp("min", terms, **clauses)


class RegExpOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        self.var, self.pattern = terms

    def __data__(self):
        return {"regexp": {self.var.var: self.pattern}}

    def vars(self):
        return {self.var.var}

    def map(self, map_):
        return RegExpOp("regex", [self.var.map(map_), self.pattern])

    def missing(self):
        return FALSE

    def exists(self):
        return TRUE


class CoalesceOp(Expression):
    has_simple_form = True

    def __init__(self, op, terms):
        Expression.__init__(self, op, terms)
        self.terms = terms

    def __data__(self):
        return {"coalesce": [t.__data__() for t in self.terms]}

    def __eq__(self, other):
        if isinstance(other, CoalesceOp):
            if len(self.terms) == len(other.terms):
                return all(s == o for s, o in zip(self.terms, other.terms))
        return False

    def missing(self):
        # RETURN true FOR RECORDS THE WOULD RETURN NULL
        return AndOp("and", [v.missing() for v in self.terms])

    def vars(self):
        output = set()
        for v in self.terms:
            output |= v.vars()
        return output

    def map(self, map_):
        return CoalesceOp("coalesce", [v.map(map_) for v in self.terms])

    @simplified
    def partial_eval(self):
        terms = []
        for t in self.terms:
            simple = FirstOp("first", t).partial_eval()
            if simple is NULL:
                pass
            elif isinstance(simple, Literal):
                terms.append(simple)
                break
            else:
                terms.append(simple)

        if len(terms) == 0:
            return NULL
        elif len(terms) == 1:
            return terms[0]
        else:
            return CoalesceOp("coalesce", terms)


class MissingOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.expr = term

    def __data__(self):
        return {"missing": self.expr.__data__()}

    def __eq__(self, other):
        if not isinstance(other, MissingOp):
            return False
        else:
            return self.expr == other.expr

    def vars(self):
        return self.expr.vars()

    def map(self, map_):
        return MissingOp("missing", self.expr.map(map_))

    def missing(self):
        return FALSE

    def exists(self):
        return TRUE

    @simplified
    def partial_eval(self):
        if isinstance(self.expr, Variable) and self.expr.var == "_id":
            return FALSE
        self.simplified = True
        return self


class ExistsOp(Expression):
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, [term])
        self.field = term

    def __data__(self):
        return {"exists": self.field.__data__()}

    def vars(self):
        return self.field.vars()

    def map(self, map_):
        return ExistsOp("exists", self.field.map(map_))

    def missing(self):
        return FALSE

    def exists(self):
        return TRUE

    @simplified
    def partial_eval(self):
        return NotOp("not", self.field.missing()).partial_eval()


class PrefixOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        if not term:
            self.field = None
            self.prefix=None
        elif isinstance(term, Mapping):
            self.field, self.prefix = term.items()[0]
        else:
            self.field, self.prefix = term

    def __data__(self):
        if not self.field:
            return {"prefix": {}}
        elif isinstance(self.field, Variable) and isinstance(self.prefix, Literal):
            return {"prefix": {self.field.var: self.prefix.value}}
        else:
            return {"prefix": [self.field.__data__(), self.prefix.__data__()]}

    def vars(self):
        if not self.field:
            return set()
        else:
            return {self.field.var}

    def map(self, map_):
        if not self.field:
            return self
        else:
            return PrefixOp("prefix", [self.field.map(map_), self.prefix.map(map_)])

    def missing(self):
        return FALSE


class SuffixOp(Expression):
    has_simple_form = True

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        if not term:
            self.field = self.suffix = None
        elif isinstance(term, Mapping):
            self.field, self.suffix = term.items()[0]
        else:
            self.field, self.suffix = term

    def __data__(self):
        if self.field is None:
            return {"suffix": {}}
        elif isinstance(self.field, Variable) and isinstance(self.suffix, Literal):
            return {"suffix": {self.field.var: self.suffix.value}}
        else:
            return {"suffix": [self.field.__data__(), self.suffix.__data__()]}

    def vars(self):
        return {self.field.var}

    def map(self, map_):
        if self.field is None:
            return TRUE
        else:
            return SuffixOp("suffix", [self.field.map(map_), self.suffix.map(map_)])


class ConcatOp(Expression):
    has_simple_form = True
    data_type = STRING

    def __init__(self, op, term, **clauses):
        Expression.__init__(self, op, term)
        if isinstance(term, Mapping):
            self.terms = term.items()[0]
        else:
            self.terms = term
        self.separator = clauses.get("separator", Literal(None, ""))
        self.default = clauses.get("default", NULL)
        if not isinstance(self.separator, Literal):
            Log.error("Expecting a literal separator")

    @classmethod
    def define(cls, expr):
        term = expr.concat
        if isinstance(term, Mapping):
            k, v = term.items()[0]
            terms = [Variable(k), Literal("literal", v)]
        else:
            terms = map(jx_expression, term)

        return ConcatOp(
            "concat",
            terms,
            **{k: Literal(None, v) for k, v in expr.items() if k in ["default", "separator"]}
        )

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.length, Literal):
            output = {"concat": {self.terms[0].var: self.terms[2].value}}
        else:
            output = {"concat": [t.__data__() for t in self.terms]}
        if self.separator.json != '""':
            output["separator"] = self.terms[2].value
        return output

    def vars(self):
        if not self.terms:
            return set()
        return set.union(*(t.vars() for t in self.terms))

    def map(self, map_):
        return ConcatOp("concat", [t.map(map_) for t in self.terms], separator=self.separator)

    def missing(self):
        return AndOp("and", [t.missing() for t in self.terms] + [self.default.missing()]).partial_eval()


class UnixOp(Expression):
    """
    FOR USING ON DATABASES WHICH HAVE A DATE COLUMNS: CONVERT TO UNIX
    """
    has_simple_form = True
    data_type = NUMBER

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.value = term

    def vars(self):
        return self.value.vars()

    def map(self, map_):
        return UnixOp("map", self.value.map(map_))

    def missing(self):
        return self.value.missing()


class FromUnixOp(Expression):
    """
    FOR USING ON DATABASES WHICH HAVE A DATE COLUMNS: CONVERT TO UNIX
    """
    data_type = NUMBER


    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.value = term

    def vars(self):
        return self.value.vars()

    def map(self, map_):
        return FromUnixOp("map", self.value.map(map_))

    def missing(self):
        return self.value.missing()


class LeftOp(Expression):
    has_simple_form = True
    data_type = STRING

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        if isinstance(term, Mapping):
            self.value, self.length = term.items()[0]
        else:
            self.value, self.length = term

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.length, Literal):
            return {"left": {self.value.var: self.length.value}}
        else:
            return {"left": [self.value.__data__(), self.length.__data__()]}

    def vars(self):
        return self.value.vars() | self.length.vars()

    def map(self, map_):
        return LeftOp("left", [self.value.map(map_), self.length.map(map_)])

    def missing(self):
        return OrOp("or", [self.value.missing(), self.length.missing()]).partial_eval()

    @simplified
    def partial_eval(self):
        value = self.value.partial_eval()
        length = self.length.partial_eval()
        max_length = LengthOp("length", value)

        return WhenOp(
            "when",
            self.missing(),
            **{
                "else": BasicSubstringOp("substring", [
                    value,
                    ZERO,
                    MaxOp("max", [ZERO, MinOp("min", [length, max_length])])
                ])
            }
        ).partial_eval()


class NotLeftOp(Expression):
    has_simple_form = True
    data_type = STRING

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        if isinstance(term, Mapping):
            self.value, self.length = term.items()[0]
        else:
            self.value, self.length = term

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.length, Literal):
            return {"not_left": {self.value.var: self.length.value}}
        else:
            return {"not_left": [self.value.__data__(), self.length.__data__()]}

    def vars(self):
        return self.value.vars() | self.length.vars()

    def map(self, map_):
        return NotLeftOp(None, [self.value.map(map_), self.length.map(map_)])

    def missing(self):
        return OrOp(None, [self.value.missing(), self.length.missing()])

    @simplified
    def partial_eval(self):
        value = self.value.partial_eval()
        length = self.length.partial_eval()
        max_length = LengthOp("length", value)

        return WhenOp(
            "when",
            self.missing(),
            **{
                "else": BasicSubstringOp("substring", [
                    value,
                    MaxOp("max", [ZERO, MinOp("min", [length, max_length])]),
                    max_length
                ])
            }
        ).partial_eval()


class RightOp(Expression):
    has_simple_form = True
    data_type = STRING

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        if isinstance(term, Mapping):
            self.value, self.length = term.items()[0]
        else:
            self.value, self.length = term

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.length, Literal):
            return {"right": {self.value.var: self.length.value}}
        else:
            return {"right": [self.value.__data__(), self.length.__data__()]}

    def vars(self):
        return self.value.vars() | self.length.vars()

    def map(self, map_):
        return RightOp("right", [self.value.map(map_), self.length.map(map_)])

    def missing(self):
        return OrOp(None, [self.value.missing(), self.length.missing()])

    @simplified
    def partial_eval(self):
        value = self.value.partial_eval()
        length = self.length.partial_eval()
        max_length = LengthOp("length", value)

        return WhenOp(
            "when",
            self.missing(),
            **{
                "else": BasicSubstringOp("substring", [
                    value,
                    MaxOp("max", [ZERO, MinOp("min", [max_length, BinaryOp("sub", [max_length, length])])]),
                    max_length
                ])
            }
        ).partial_eval()


class NotRightOp(Expression):
    has_simple_form = True
    data_type = STRING

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        if isinstance(term, Mapping):
            self.value, self.length = term.items()[0]
        else:
            self.value, self.length = term

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.length, Literal):
            return {"not_right": {self.value.var: self.length.value}}
        else:
            return {"not_right": [self.value.__data__(), self.length.__data__()]}

    def vars(self):
        return self.value.vars() | self.length.vars()

    def map(self, map_):
        return NotRightOp(None, [self.value.map(map_), self.length.map(map_)])

    def missing(self):
        return OrOp(None, [self.value.missing(), self.length.missing()])

    @simplified
    def partial_eval(self):
        value = self.value.partial_eval()
        length = self.length.partial_eval()
        max_length = LengthOp("length", value)

        return WhenOp(
            "when",
            self.missing(),
            **{
                "else": BasicSubstringOp("substring", [
                    value,
                    ZERO,
                    MaxOp("max", [ZERO, MinOp("min", [max_length, BinaryOp("sub", [max_length, length])])])
                ])
            }
        ).partial_eval()


class FindOp(Expression):
    """
    RETURN INDEX OF find IN value, ELSE RETURN null
    """
    has_simple_form = True
    data_type = INTEGER

    def __init__(self, op, term, **kwargs):
        Expression.__init__(self, op, term)
        self.value, self.find = term
        self.default = kwargs.get("default", NULL)
        self.start = kwargs.get("start", ZERO).partial_eval()
        if isinstance(self.start, NullOp):
            self.start = ZERO

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.find, Literal):
            output = {
                "find": {self.value.var, self.find.value},
                "start":self.start.__data__()
            }
        else:
            output = {
                "find": [self.value.__data__(), self.find.__data__()],
                "start":self.start.__data__()
            }
        if self.default:
            output["default"]=self.default.__data__()
        return output

    def vars(self):
        return self.value.vars() | self.find.vars() | self.default.vars() | self.start.vars()

    def map(self, map_):
        return FindOp(
            "find",
            [self.value.map(map_), self.find.map(map_)],
            start=self.start.map(map_),
            default=self.default.map(map_)
        )

    def missing(self):
        return AndOp("and", [
            self.default.missing(),
            OrOp("or", [
                self.value.missing(),
                self.find.missing(),
                EqOp("eq", [BasicIndexOfOp("", [
                    self.value,
                    self.find,
                    self.start
                ]), Literal(None, -1)])
            ])
        ]).partial_eval()

    def exists(self):
        return TRUE

    @simplified
    def partial_eval(self):
        index = BasicIndexOfOp("indexOf", [
            self.value,
            self.find,
            self.start
        ]).partial_eval()

        output = WhenOp(
            "when",
            OrOp("or", [
                self.value.missing(),
                self.find.missing(),
                BasicEqOp("eq", [index, Literal(None, -1)])
            ]),
            **{"then": self.default, "else": index}
        ).partial_eval()
        return output


class SplitOp(Expression):
    has_simple_form = True

    def __init__(self, op, term, **kwargs):
        Expression.__init__(self, op, term)
        self.value, self.find = term

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.find, Literal):
            return {"split": {self.value.var, self.find.value}}
        else:
            return {"split": [self.value.__data__(), self.find.__data__()]}

    def vars(self):
        return self.value.vars() | self.find.vars() | self.default.vars() | self.start.vars()

    def map(self, map_):
        return FindOp(
            "find",
            [self.value.map(map_), self.find.map(map_)],
            start=self.start.map(map_),
            default=self.default.map(map_)
        )

    def missing(self):
        v = self.value.to_ruby(not_null=True)
        find = self.find.to_ruby(not_null=True)
        index = v + ".indexOf(" + find + ", " + self.start.to_ruby() + ")"

        return AndOp("and", [
            self.default.missing(),
            OrOp("or", [
                self.value.missing(),
                self.find.missing(),
                EqOp("eq", [ScriptOp("script", index), Literal(None, -1)])
            ])
        ])

    def exists(self):
        return TRUE


class BetweenOp(Expression):
    data_type = STRING

    def __init__(self, op, value, prefix, suffix, default=NULL, start=NULL):
        Expression.__init__(self, op, [])
        self.value = value
        self.prefix = prefix
        self.suffix = suffix
        self.default = default
        self.start = start
        if isinstance(self.prefix, Literal) and isinstance(self.suffix, Literal):
            pass
        else:
            Log.error("Expecting literal prefix and suffix only")

    @classmethod
    def define(cls, expr):
        term = expr.between
        if isinstance(term, list):
            return BetweenOp(
                "between",
                value=jx_expression(term[0]),
                prefix=jx_expression(term[1]),
                suffix=jx_expression(term[2]),
                default=jx_expression(expr.default),
                start=jx_expression(expr.start)
            )
        elif isinstance(term, Mapping):
            var, vals = term.items()[0]
            if isinstance(vals, list) and len(vals) == 2:
                return BetweenOp(
                    "between",
                    value=Variable(var),
                    prefix=Literal(None, vals[0]),
                    suffix=Literal(None, vals[1]),
                    default=jx_expression(expr.default),
                    start=jx_expression(expr.start)
                )
            else:
                Log.error("`between` parameters are expected to be in {var: [prefix, suffix]} form")
        else:
            Log.error("`between` parameters are expected to be in {var: [prefix, suffix]} form")

    def vars(self):
        return self.value.vars() | self.prefix.vars() | self.suffix.vars() | self.default.vars() | self.start.vars()

    def map(self, map_):
        return BetweenOp(
            "between",
            [self.value.map(map_), self.prefix.map(map_), self.suffix.map(map_)],
            default=self.default.map(map_),
            start=self.start.map(map_)
        )

    def missing(self):
        return self.partial_eval().missing()

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.prefix, Literal) and isinstance(self.suffix, Literal):
            output = wrap({"between": {self.value.var: [self.prefix.value, self.suffix.value]}})
        else:
            output = wrap({"between": [self.value.__data__(), self.prefix.__data__(), self.suffix.__data__()]})
        if self.start:
            output.start = self.start.__data__()
        if self.default:
            output.default = self.default.__data__()
        return output

    @simplified
    def partial_eval(self):
        value = self.value.partial_eval()

        start_index = CaseOp(
            "case",
            [
                WhenOp("when", self.prefix.missing(), **{"then": ZERO}),
                WhenOp("when", IsNumberOp("is_number", self.prefix), **{"then": MaxOp("max", [ZERO, self.prefix])}),
                FindOp("find", [value, self.prefix], start=self.start)
            ]
        ).partial_eval()

        len_prefix = CaseOp(
            "case",
            [
                WhenOp("when", self.prefix.missing(), **{"then": ZERO}),
                WhenOp("when", IsNumberOp("is_number", self.prefix), **{"then": ZERO}),
                LengthOp("length", self.prefix)
            ]
        ).partial_eval()

        end_index = CaseOp(
            "case",
            [
                WhenOp("when", start_index.missing(), **{"then": NULL}),
                WhenOp("when", self.suffix.missing(), **{"then": LengthOp("length", value)}),
                WhenOp("when", IsNumberOp("is_number", self.suffix), **{"then": MinOp("min", [self.suffix, LengthOp("length", value)])}),
                FindOp("find", [value, self.suffix], start=MultiOp("add", [start_index, len_prefix]))
            ]
        ).partial_eval()

        start_index = MultiOp("add", [start_index, len_prefix]).partial_eval()
        substring = BasicSubstringOp("subtring", [value, start_index, end_index]).partial_eval()

        between = WhenOp(
            "when",
            end_index.missing(),
            **{
                "then": self.default,
                "else": substring
            }
        ).partial_eval()

        return between


class InOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    def __init__(self, op, term):
        Expression.__init__(self, op, term)
        self.value, self.superset = term

    def __data__(self):
        if isinstance(self.value, Variable) and isinstance(self.superset, Literal):
            return {"in": {self.value.var: self.superset.value}}
        else:
            return {"in": [self.value.__data__(), self.superset.__data__()]}

    def __eq__(self, other):
        if isinstance(other, InOp):
            return self.value == other.value and self.superset == other.superset
        return False

    def vars(self):
        return self.value.vars()

    def map(self, map_):
        return InOp("in", [self.value.map(map_), self.superset.map(map_)])

    @simplified
    def partial_eval(self):
        value = self.value.partial_eval()
        superset = self.superset.partial_eval()
        if isinstance(value, Literal) and isinstance(superset, Literal):
            return Literal(None, self())
        else:
            return self

    def __call__(self):
        return self.value() in self.superset()

    def missing(self):
        return FALSE


class RangeOp(Expression):
    has_simple_form = True
    data_type = BOOLEAN

    def __new__(cls, op, term, *args):
        Expression.__new__(cls, *args)
        field, comparisons = term  # comparisons IS A Literal()
        return AndOp("and", [operators[op](op, [field, Literal(None, value)]) for op, value in comparisons.value.items()])

    def __init__(self, op, term):
        Log.error("Should never happen!")


class WhenOp(Expression):
    def __init__(self, op, term, **clauses):
        Expression.__init__(self, op, [term])

        self.when = term
        self.then = coalesce(clauses.get("then"), NULL)
        self.els_ = coalesce(clauses.get("else"), NULL)

        if self.then is NULL:
            self.data_type = self.els_.type
        elif self.els_ is NULL:
            self.data_type = self.then.type
        elif self.then.type == self.els_.type:
            self.data_type = self.then.type
        elif self.then.type in (INTEGER, NUMBER) and self.els_.type in (INTEGER, NUMBER):
            self.data_type = NUMBER
        else:
            self.data_type = OBJECT

    def __data__(self):
        return {"when": self.when.__data__(), "then": self.then.__data__() if self.then else None, "else": self.els_.__data__() if self.els_ else None}

    def vars(self):
        return self.when.vars() | self.then.vars() | self.els_.vars()

    def map(self, map_):
        return WhenOp("when", self.when.map(map_), **{"then": self.then.map(map_), "else": self.els_.map(map_)})

    def missing(self):
        return OrOp("or", [
            AndOp("and", [self.when, self.then.missing()]),
            AndOp("and", [NotOp("not", self.when), self.els_.missing()])
        ]).partial_eval()

    @simplified
    def partial_eval(self):
        when = BooleanOp("boolean", self.when).partial_eval()

        if when is TRUE:
            return self.then.partial_eval()
        elif when in [FALSE, NULL]:
            return self.els_.partial_eval()
        elif isinstance(when, Literal):
            Log.error("Expecting `when` clause to return a Boolean, or `null`")

        then = self.then.partial_eval()
        els_ = self.els_.partial_eval()

        if then is TRUE:
            if els_ is FALSE:
                return when
            elif els_ is TRUE:
                return TRUE
        elif then is FALSE:
            if els_ is FALSE:
                return FALSE
            elif els_ is TRUE:
                return NotOp("not", when).partial_eval()

        return WhenOp("when", when, **{"then": then, "else": els_})


class CaseOp(Expression):
    def __init__(self, op, term, **clauses):
        if not isinstance(term, (list, tuple)):
            Log.error("case expression requires a list of `when` sub-clauses")
        Expression.__init__(self, op, term)
        if len(term) <= 1:
            Log.error("Expecting at least two clauses")
        else:
            for w in term[:-1]:
                if not isinstance(w, WhenOp) or w.els_:
                    Log.error("case expression does not allow `else` clause in `when` sub-clause")
            self.whens = term

    def __data__(self):
        return {"case": [w.__data__() for w in self.whens]}

    def vars(self):
        output = set()
        for w in self.whens:
            output |= w.vars()
        return output

    def map(self, map_):
        return CaseOp("case", [w.map(map_) for w in self.whens])

    def missing(self):
        m = self.whens[-1].missing()
        for w in reversed(self.whens[0:-1]):
            when = w.when.partial_eval()
            if when is FALSE:
                pass
            elif when is TRUE:
                m = w.then.partial_eval().missing()
            else:
                m = OrOp("or", [AndOp("and", [when, w.then.partial_eval().missing()]), m])
        return m.partial_eval()

    @simplified
    def partial_eval(self):
        whens = []
        for w in self.whens[:-1]:
            when = w.when.partial_eval()
            if when is TRUE:
                whens.append(w.then.partial_eval())
                break
            elif when is FALSE:
                pass
            else:
                whens.append(WhenOp("when", when, **{"then": w.then.partial_eval()}))
        else:
            whens.append(self.whens[-1].partial_eval())

        if len(whens) == 1:
            return whens[0]
        else:
            return CaseOp("case", whens)



class BasicIndexOfOp(Expression):
    """
    PLACEHOLDER FOR BASIC value.indexOf(find, start) (CAN NOT DEAL WITH NULLS)
    """
    data_type = INTEGER

    def __init__(self, op, params):
        Expression.__init__(self, op, params)
        self.value, self.find, self.start = params

    def __data__(self):
        return {"basic.indexOf": [self.value.__data__(), self.find.__data__(), self.start.__data__()]}

    def vars(self):
        return self.value.vars() | self.find.vars() | self.start.vars()

    def missing(self):
        return FALSE

    @simplified
    def partial_eval(self):
        start = IntegerOp("integer", MaxOp("max", [ZERO, self.start])).partial_eval()
        return BasicIndexOfOp("indexOf", [
            StringOp("string", self.value).partial_eval(),
            StringOp("string", self.find).partial_eval(),
            start
        ])


class BasicEqOp(Expression):
    """
    PLACEHOLDER FOR BASIC `==` OPERATOR (CAN NOT DEAL WITH NULLS)
    """
    data_type = BOOLEAN

    def __init__(self, op, terms):
        self.lhs, self.rhs = terms

    def __data__(self):
        return {"basic.eq": [self.lhs.__data__(), self.rhs.__data__()]}

    def missing(self):
        return FALSE


class BasicSubstringOp(Expression):
    """
    PLACEHOLDER FOR BASIC value.substring(start, end) (CAN NOT DEAL WITH NULLS)
    """
    data_type = STRING

    def __init__(self, op, terms):
        self.value, self.start, self.end = terms

    def __data__(self):
        return {"basic.substring": [self.value.__data__(), self.start.__data__(), self.end.__data__()]}

    def missing(self):
        return FALSE



operators = {
    "add": MultiOp,
    "and": AndOp,
    "between": BetweenOp,
    "case": CaseOp,
    "coalesce": CoalesceOp,
    "concat": ConcatOp,
    "count": CountOp,
    "date": DateOp,
    "div": DivOp,
    "divide": DivOp,
    "eq": EqOp,
    "exists": ExistsOp,
    "exp": BinaryOp,
    "find": FindOp,
    "first": FirstOp,
    "floor": FloorOp,
    "from_unix": FromUnixOp,
    "get": GetOp,
    "gt": InequalityOp,
    "gte": InequalityOp,
    "in": InOp,
    "instr": FindOp,
    "is_number": IsNumberOp,
    "is_string": IsStringOp,
    "left": LeftOp,
    "length": LengthOp,
    "literal": Literal,
    "lt": InequalityOp,
    "lte": InequalityOp,
    "match_all": TrueOp,
    "max": MaxOp,
    "minus": BinaryOp,
    "missing": MissingOp,
    "mod": BinaryOp,
    "mul": MultiOp,
    "mult": MultiOp,
    "multiply": MultiOp,
    "ne": NeOp,
    "neq": NeOp,
    "not": NotOp,
    "not_left": NotLeftOp,
    "not_right": NotRightOp,
    "null": NullOp,
    "number": NumberOp,
    "offset": OffsetOp,
    "or": OrOp,
    "postfix": SuffixOp,
    "prefix": PrefixOp,
    "range": RangeOp,
    "regex": RegExpOp,
    "regexp": RegExpOp,
    "right": RightOp,
    "rows": RowsOp,
    "script": ScriptOp,
    "select": SelectOp,
    "split": SplitOp,
    "string": StringOp,
    "suffix": SuffixOp,
    "sub": BinaryOp,
    "subtract": BinaryOp,
    "sum": MultiOp,
    "term": EqOp,
    "terms": InOp,
    "tuple": TupleOp,
    "unix": UnixOp,
    "when": WhenOp,
}


builtin_ops = {
    "ne": operator.ne,
    "eq": operator.eq,
    "gte": operator.ge,
    "gt": operator.gt,
    "lte": operator.le,
    "lt": operator.lt,
    "add": operator.add,
    "sum": operator.add,
    "mul": operator.mul,
    "mult": operator.mul,
    "multiply": operator.mul,
    "max": lambda *v: max(v),
    "min": lambda *v: min(v)
}

# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

from jx_base.expressions import (
    AndOp as AndOp_,
    BaseBinaryOp as BaseBinaryOp_,
    BaseInequalityOp as BaseInequalityOp_,
    BaseMultiOp as BaseMultiOp_,
    BasicEqOp as BasicEqOp_,
    BasicIndexOfOp as BasicIndexOfOp_,
    BetweenOp as BetweenOp_,
    CaseOp as CaseOp_,
    CoalesceOp as CoalesceOp_,
    ConcatOp as ConcatOp_,
    CountOp as CountOp_,
    DateOp as DateOp_,
    DivOp as DivOp_,
    EqOp as EqOp_,
    ExistsOp as ExistsOp_,
    ExpOp as ExpOp_,
    FALSE,
    FalseOp as FalseOp_,
    FindOp as FindOp_,
    FirstOp as FirstOp_,
    FloorOp as FloorOp_,
    GetOp as GetOp_,
    GtOp as GtOp_,
    GteOp as GteOp_,
    InOp as InOp_,
    IntegerOp as IntegerOp_,
    LastOp as LastOp_,
    LeavesOp as LeavesOp_,
    LengthOp as LengthOp_,
    Literal as Literal_,
    LtOp as LtOp_,
    LteOp as LteOp_,
    MaxOp as MaxOp_,
    MissingOp as MissingOp_,
    ModOp as ModOp_,
    NULL,
    NeOp as NeOp_,
    NotLeftOp as NotLeftOp_,
    NotOp as NotOp_,
    NotRightOp as NotRightOp_,
    NullOp,
    NumberOp as NumberOp_,
    ONE,
    OffsetOp as OffsetOp_,
    OrOp as OrOp_,
    PrefixOp as PrefixOp_,
    PythonScript as PythonScript_,
    RangeOp as RangeOp_,
    RegExpOp as RegExpOp_,
    RightOp as RightOp_,
    RowsOp as RowsOp_,
    ScriptOp as ScriptOp_,
    SelectOp as SelectOp_,
    SplitOp as SplitOp_,
    StringOp as StringOp_,
    SubOp as SubOp_,
    SuffixOp as SuffixOp_,
    TRUE,
    TrueOp as TrueOp_,
    TupleOp as TupleOp_,
    Variable as Variable_,
    WhenOp as WhenOp_,
    ZERO,
    define_language,
    extend,
    jx_expression,
)
from jx_base.language import is_expression, is_op
from jx_python.expression_compiler import compile_expression
from mo_dots import coalesce, is_data, is_list, split_field, unwrap
from mo_future import PY2, is_text, text_type
from mo_json import BOOLEAN, INTEGER, NUMBER, json2value
from mo_logs import Log
from mo_logs.strings import quote
from mo_times.dates import Date
from pyLibrary import convert


def jx_expression_to_function(expr):
    """
    RETURN FUNCTION THAT REQUIRES PARAMETERS (row, rownum=None, rows=None):
    """
    if is_expression(expr):
        if is_op(expr, ScriptOp) and not is_text(expr.script):
            return expr.script
        else:
            return compile_expression(Python[expr].to_python())
    if (
        expr != None
        and not is_data(expr)
        and not is_list(expr)
        and hasattr(expr, "__call__")
    ):
        return expr
    return compile_expression(Python[jx_expression(expr)].to_python())


class PythonScript(PythonScript_):
    __slots__ = ("miss", "data_type", "expr", "many")

    def __init__(self, type, expr, frum, miss=None, many=False):
        object.__init__(self)
        if miss not in [None, NULL, FALSE, TRUE, ONE, ZERO]:
            if frum.lang != miss.lang:
                Log.error("logic error")

        self.miss = coalesce(
            miss, FALSE
        )  # Expression that will return true/false to indicate missing result
        self.data_type = type
        self.expr = expr
        self.many = many  # True if script returns multi-value
        self.frum = frum  # THE ORIGINAL EXPRESSION THAT MADE expr

    @property
    def type(self):
        return self.data_type

    def __str__(self):
        """
        RETURN A SCRIPT SUITABLE FOR CODE OUTSIDE THIS MODULE (NO KNOWLEDGE OF Painless)
        :param schema:
        :return:
        """
        missing = self.miss.partial_eval()
        if missing is FALSE:
            return self.partial_eval().to_python().expr
        elif missing is TRUE:
            return "None"

        return "None if (" + missing.to_python().expr + ") else (" + self.expr + ")"

    def __add__(self, other):
        return text_type(self) + text_type(other)

    def __radd__(self, other):
        return text_type(other) + text_type(self)

    if PY2:
        __unicode__ = __str__

    def to_python(self, not_null=False, boolean=False, many=True):
        return self

    def missing(self):
        return self.miss

    def __data__(self):
        return {"script": self.script}

    def __eq__(self, other):
        if not isinstance(other, PythonScript_):
            return False
        elif self.expr == other.expr:
            return True
        else:
            return False


class Variable(Variable_):
    def to_python(self, not_null=False, boolean=False, many=False):
        path = split_field(self.var)
        agg = "row"
        if not path:
            return agg
        elif path[0] in ["row", "rownum"]:
            # MAGIC VARIABLES
            agg = path[0]
            path = path[1:]
            if len(path) == 0:
                return agg
        elif path[0] == "rows":
            if len(path) == 1:
                return "rows"
            elif path[1] in ["first", "last"]:
                agg = "rows." + path[1] + "()"
                path = path[2:]
            else:
                Log.error("do not know what {{var}} of `rows` is", var=path[1])

        for p in path[:-1]:
            if not_null:
                agg = agg + ".get(" + convert.value2quote(p) + ")"
            else:
                agg = agg + ".get(" + convert.value2quote(p) + ", EMPTY_DICT)"
        output = agg + ".get(" + convert.value2quote(path[-1]) + ")"
        if many:
            output = "listwrap(" + output + ")"
        return output


class OffsetOp(OffsetOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "row["
            + text_type(self.var)
            + "] if 0<="
            + text_type(self.var)
            + "<len(row) else None"
        )


class RowsOp(RowsOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        agg = "rows[rownum+" + Python[IntegerOp(self.offset)].to_python() + "]"
        path = split_field(json2value(self.var.json))
        if not path:
            return agg

        for p in path[:-1]:
            agg = agg + ".get(" + convert.value2quote(p) + ", EMPTY_DICT)"
        return agg + ".get(" + convert.value2quote(path[-1]) + ")"


class IntegerOp(IntegerOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "int(" + Python[self.term].to_python() + ")"


class GetOp(GetOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        obj = Python[self.var].to_python()
        code = Python[self.offset].to_python()
        return "listwrap(" + obj + ")[" + code + "]"


class LastOp(LastOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        term = Python[self.term].to_python()
        return "listwrap(" + term + ").last()"


class SelectOp(SelectOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "wrap_leaves({"
            + ",".join(
                quote(t["name"]) + ":" + Python[t["value"]].to_python()
                for t in self.terms
            )
            + "})"
        )


class ScriptOp(ScriptOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return self.script


class Literal(Literal_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return text_type(repr(unwrap(json2value(self.json))))


@extend(NullOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "None"


class TrueOp(TrueOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "True"


class FalseOp(FalseOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "False"


class DateOp(DateOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return text_type(Date(self.value).unix)


class TupleOp(TupleOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        if len(self.terms) == 0:
            return "tuple()"
        elif len(self.terms) == 1:
            return "(" + Python[self.terms[0]].to_python() + ",)"
        else:
            return "(" + (",".join(Python[t].to_python() for t in self.terms)) + ")"


class LeavesOp(LeavesOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "Data(" + Python[self.term].to_python() + ").leaves()"


def _inequality_to_python(self, not_null=False, boolean=False, many=True):
    op, identity = _python_operators[self.op]
    lhs = NumberOp(self.lhs).partial_eval().to_python(not_null=True)
    rhs = NumberOp(self.rhs).partial_eval().to_python(not_null=True)
    script = "(" + lhs + ") " + op + " (" + rhs + ")"

    output = (
        WhenOp(
            OrOp([self.lhs.missing(), self.rhs.missing()]),
            **{
                "then": FALSE,
                "else": PythonScript(type=BOOLEAN, expr=script, frum=self),
            }
        )
        .partial_eval()
        .to_python()
    )
    return output


class GtOp(GtOp_):
    to_python = _inequality_to_python


class GteOp(GteOp_):
    to_python = _inequality_to_python


class LtOp(LtOp_):
    to_python = _inequality_to_python


class LteOp(LteOp_):
    to_python = _inequality_to_python


class BaseBinaryOp(BaseBinaryOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.lhs].to_python()
            + ") "
            + _python_operators[self.op][0]
            + " ("
            + Python[self.rhs].to_python()
            + ")"
        )


class BaseInequalityOp(BaseInequalityOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.lhs].to_python()
            + ") "
            + _python_operators[self.op][0]
            + " ("
            + Python[self.rhs].to_python()
            + ")"
        )


class InOp(InOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            Python[self.value].to_python()
            + " in "
            + Python[self.superset].to_python(many=True)
        )


class DivOp(DivOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        miss = Python[self.missing()].to_python()
        lhs = Python[self.lhs].to_python(not_null=True)
        rhs = Python[self.rhs].to_python(not_null=True)
        return "None if (" + miss + ") else (" + lhs + ") / (" + rhs + ")"


class FloorOp(FloorOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "mo_math.floor("
            + Python[self.lhs].to_python()
            + ", "
            + Python[self.rhs].to_python()
            + ")"
        )


class EqOp(EqOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.rhs].to_python()
            + ") in listwrap("
            + Python[self.lhs].to_python()
            + ")"
        )


class BasicEqOp(BasicEqOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.rhs].to_python()
            + ") == ("
            + Python[self.lhs].to_python()
            + ")"
        )


class NeOp(NeOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        lhs = Python[self.lhs].to_python()
        rhs = Python[self.rhs].to_python()
        return (
            "(("
            + lhs
            + ") != None and ("
            + rhs
            + ") != None and ("
            + lhs
            + ") != ("
            + rhs
            + "))"
        )


class NotOp(NotOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "not (" + Python[self.term].to_python(boolean=True) + ")"


class AndOp(AndOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        if not self.terms:
            return "True"
        else:
            return " and ".join("(" + Python[t].to_python() + ")" for t in self.terms)


class OrOp(OrOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return " or ".join("(" + Python[t].to_python() + ")" for t in self.terms)


class LengthOp(LengthOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        value = Python[self.term].to_python()
        return "len(" + value + ") if (" + value + ") != None else None"


class FirstOp(FirstOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        value = Python[self.term].to_python()
        return "listwrap(" + value + ").first()"


class NumberOp(NumberOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        term = Python[self.term]
        if not_null:
            if term.type in [NUMBER, INTEGER]:
                return term.to_python(not_null=True)
            else:
                return "float(" + Python[self.term].to_python(not_null=True) + ")"
        else:
            exists = self.term.exists()
            value = Python[self.term].to_python(not_null=True)

            if exists is TRUE:
                return "float(" + value + ")"
            else:
                return (
                    "float("
                    + value
                    + ") if ("
                    + Python[exists].to_python()
                    + ") else None"
                )


class StringOp(StringOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        missing = Python[self.term.missing()].to_python(boolean=True)
        value = Python[self.term].to_python(not_null=True)
        return "null if (" + missing + ") else text_type(" + value + ")"


class CountOp(CountOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "+".join(
            "(0 if (" + Python[t.missing()].to_python(boolean=True) + ") else 1)"
            for t in self.terms
        )


class MaxOp(MaxOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return "max([" + (",".join(Python[t].to_python() for t in self.terms)) + "])"


class BaseMultiOp(BaseMultiOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        sign, zero = _python_operators[self.op]
        if len(self.terms) == 0:
            return Python[self.default].to_python()
        elif self.default is NULL:
            return sign.join(
                "coalesce(" + Python[t].to_python() + ", " + zero + ")"
                for t in self.terms
            )
        else:
            return (
                "coalesce("
                + sign.join("(" + Python[t].to_python() + ")" for t in self.terms)
                + ", "
                + Python[self.default].to_python()
                + ")"
            )


class RegExpOp(RegExpOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "re.match("
            + quote(json2value(self.pattern.json) + "$")
            + ", "
            + Python[self.var].to_python()
            + ")"
        )


class CoalesceOp(CoalesceOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "coalesce(" + (", ".join(Python[t].to_python() for t in self.terms)) + ")"
        )


class MissingOp(MissingOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return Python[self.expr].to_python() + " == None"


class ExistsOp(ExistsOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return Python[self.field].to_python() + " != None"


class PrefixOp(PrefixOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.expr].to_python()
            + ").startswith("
            + Python[self.prefix].to_python()
            + ")"
        )


class SuffixOp(SuffixOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.expr].to_python()
            + ").endswith("
            + Python[self.suffix].to_python()
            + ")"
        )


class ConcatOp(ConcatOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        v = Python[self.value].to_python()
        l = Python[self.length].to_python()
        return (
            "None if "
            + v
            + " == None or "
            + l
            + " == None else "
            + v
            + "[0:max(0, "
            + l
            + ")]"
        )


class NotLeftOp(NotLeftOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        v = Python[self.value].to_python()
        l = Python[self.length].to_python()
        return (
            "None if "
            + v
            + " == None or "
            + l
            + " == None else "
            + v
            + "[max(0, "
            + l
            + "):]"
        )


class RightOp(RightOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        v = Python[self.value].to_python()
        l = Python[self.length].to_python()
        return (
            "None if "
            + v
            + " == None or "
            + l
            + " == None else "
            + v
            + "[max(0, len("
            + v
            + ")-("
            + l
            + ")):]"
        )


class NotRightOp(NotRightOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        v = Python[self.value].to_python()
        l = Python[self.length].to_python()
        return (
            "None if "
            + v
            + " == None or "
            + l
            + " == None else "
            + v
            + "[0:max(0, len("
            + v
            + ")-("
            + l
            + "))]"
        )


class BasicIndexOfOp(BasicIndexOfOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        find = (
            "("
            + Python[self.value].to_python()
            + ").find("
            + Python[self.find].to_python()
            + ", "
            + Python[self.start].to_python()
            + ")"
        )
        return "[None if i==-1 else i for i in [" + find + "]][0]"


class SplitOp(SplitOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.value].to_python()
            + ").split("
            + Python[self.find].to_python()
            + ")"
        )


class FindOp(FindOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "(("
            + quote(self.substring)
            + " in "
            + Python[self.var].to_python()
            + ") if "
            + Python[self.var].to_python()
            + "!=None else False)"
        )


class BetweenOp(BetweenOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            Python[self.value].to_python()
            + " in "
            + Python[self.superset].to_python(many=True)
        )


class RangeOp(RangeOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.then].to_python(not_null=not_null)
            + ") if ("
            + Python[self.when].to_python(boolean=True)
            + ") else ("
            + Python[self.els_].to_python(not_null=not_null)
            + ")"
        )


def _binary_to_python(self, not_null=False, boolean=False, many=True):
    op, identity = _python_operators[self.op]

    lhs = NumberOp(self.lhs).partial_eval().to_python(not_null=True)
    rhs = NumberOp(self.rhs).partial_eval().to_python(not_null=True)
    script = "(" + lhs + ") " + op + " (" + rhs + ")"
    missing = OrOp([self.lhs.missing(), self.rhs.missing()]).partial_eval()
    if missing is FALSE:
        return script
    else:
        return "(None) if (" + missing.to_python() + ") else (" + script + ")"


class SubOp(SubOp_):
    to_python = _binary_to_python


class ExpOp(ExpOp_):
    to_python = _binary_to_python


class ModOp(ModOp_):
    to_python = _binary_to_python


class CaseOp(CaseOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        acc = Python[self.whens[-1]].to_python()
        for w in reversed(self.whens[0:-1]):
            acc = (
                "("
                + Python[w.then].to_python()
                + ") if ("
                + Python[w.when].to_python(boolean=True)
                + ") else ("
                + acc
                + ")"
            )
        return acc


class WhenOp(WhenOp_):
    def to_python(self, not_null=False, boolean=False, many=False):
        return (
            "("
            + Python[self.then].to_python()
            + ") if ("
            + Python[self.when].to_python(boolean=True)
            + ") else ("
            + Python[self.els_].to_python()
            + ")"
        )


Python = define_language("Python", vars())


_python_operators = {
    "add": (" + ", "0"),  # (operator, zero-array default value) PAIR
    "sum": (" + ", "0"),
    "mul": (" * ", "1"),
    "sub": (" - ", None),
    "div": (" / ", None),
    "exp": (" ** ", None),
    "mod": (" % ", None),
    "gt": (" > ", None),
    "gte": (" >= ", None),
    "lte": (" <= ", None),
    "lt": (" < ", None),
}

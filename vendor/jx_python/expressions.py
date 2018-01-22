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

from collections import Mapping

from mo_future import text_type
from mo_dots import split_field
from mo_dots import unwrap
from mo_json import json2value
from mo_logs import Log
from mo_logs.strings import quote
from pyLibrary import convert

from jx_base.expressions import Variable, DateOp, TupleOp, LeavesOp, BinaryOp, OrOp, ScriptOp, \
    InequalityOp, extend, RowsOp, OffsetOp, GetOp, Literal, NullOp, TrueOp, FalseOp, DivOp, FloorOp, \
    EqOp, NeOp, NotOp, LengthOp, NumberOp, StringOp, CountOp, MultiOp, RegExpOp, CoalesceOp, MissingOp, ExistsOp, \
    PrefixOp, NotLeftOp, RightOp, NotRightOp, FindOp, BetweenOp, RangeOp, CaseOp, AndOp, \
    ConcatOp, InOp, jx_expression, Expression, WhenOp, MaxOp, SplitOp, NULL, SelectOp, SuffixOp
from jx_python.expression_compiler import compile_expression
from mo_times.dates import Date


def jx_expression_to_function(expr):
    """
    RETURN FUNCTION THAT REQUIRES PARAMETERS (row, rownum=None, rows=None):
    """
    if isinstance(expr, Expression):
        if isinstance(expr, ScriptOp) and not isinstance(expr.script, text_type):
            return expr.script
        else:
            return compile_expression(expr.to_python())
    if expr != None and not isinstance(expr, (Mapping, list)) and hasattr(expr, "__call__"):
        return expr
    return compile_expression(jx_expression(expr).to_python())


@extend(Variable)
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


@extend(OffsetOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "row[" + text_type(self.var) + "] if 0<=" + text_type(self.var) + "<len(row) else None"


@extend(RowsOp)
def to_python(self, not_null=False, boolean=False, many=False):
    agg = "rows[rownum+" + self.offset.to_python() + "]"
    path = split_field(json2value(self.var.json))
    if not path:
        return agg

    for p in path[:-1]:
        agg = agg + ".get(" + convert.value2quote(p) + ", EMPTY_DICT)"
    return agg + ".get(" + convert.value2quote(path[-1]) + ")"


@extend(GetOp)
def to_python(self, not_null=False, boolean=False, many=False):
    obj = self.var.to_python()
    code = self.offset.to_python()
    return "listwrap("+obj+")[" + code + "]"


@extend(SelectOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return (
        "wrap_leaves({" +
        ",\n".join(
            quote(t['name']) + ":" + t['value'].to_python() for t in self.terms
        ) +
        "})"
    )


@extend(ScriptOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return self.script


@extend(Literal)
def to_python(self, not_null=False, boolean=False, many=False):
    return text_type(repr(unwrap(json2value(self.json))))


@extend(NullOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "None"


@extend(TrueOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "True"


@extend(FalseOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "False"


@extend(DateOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return text_type(Date(self.value).unix)


@extend(TupleOp)
def to_python(self, not_null=False, boolean=False, many=False):
    if len(self.terms) == 0:
        return "tuple()"
    elif len(self.terms) == 1:
        return "(" + self.terms[0].to_python() + ",)"
    else:
        return "(" + (",".join(t.to_python() for t in self.terms)) + ")"


@extend(LeavesOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "Data(" + self.term.to_python() + ").leaves()"


@extend(BinaryOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.lhs.to_python() + ") " + BinaryOp.operators[self.op] + " (" + self.rhs.to_python() + ")"


@extend(InequalityOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.lhs.to_python() + ") " + InequalityOp.operators[self.op] + " (" + self.rhs.to_python() + ")"


@extend(InOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return self.value.to_python() + " in " + self.superset.to_python(many=True)


@extend(DivOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "None if (" + self.missing().to_python() + ") else (" + self.lhs.to_python(
        not_null=True) + ") / (" + self.rhs.to_python(not_null=True) + ")"


@extend(FloorOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "Math.floor(" + self.lhs.to_python() + ", " + self.rhs.to_python() + ")"


@extend(EqOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.rhs.to_python() + ") in listwrap(" + self.lhs.to_python() + ")"


@extend(NeOp)
def to_python(self, not_null=False, boolean=False, many=False):
    lhs = self.lhs.to_python()
    rhs = self.rhs.to_python()
    return "((" + lhs + ") != None and (" + rhs + ") != None and (" + lhs + ") != (" + rhs + "))"


@extend(NotOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "not (" + self.term.to_python(boolean=True) + ")"


@extend(AndOp)
def to_python(self, not_null=False, boolean=False, many=False):
    if not self.terms:
        return "True"
    else:
        return " and ".join("(" + t.to_python() + ")" for t in self.terms)


@extend(OrOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return " or ".join("(" + t.to_python() + ")" for t in self.terms)


@extend(LengthOp)
def to_python(self, not_null=False, boolean=False, many=False):
    value = self.term.to_python()
    return "len(" + value + ") if (" + value + ") != None else None"


@extend(NumberOp)
def to_python(self, not_null=False, boolean=False, many=False):
    test = self.term.missing().to_python(boolean=True)
    value = self.term.to_python(not_null=True)
    return "float(" + value + ") if (" + test + ") else None"


@extend(StringOp)
def to_python(self, not_null=False, boolean=False, many=False):
    missing = self.term.missing().to_python(boolean=True)
    value = self.term.to_python(not_null=True)
    return "null if (" + missing + ") else text_type(" + value + ")"


@extend(CountOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "+".join("(0 if (" + t.missing().to_python(boolean=True) + ") else 1)" for t in self.terms)


@extend(MaxOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "max(["+(",".join(t.to_python() for t in self.terms))+"])"


@extend(MultiOp)
def to_python(self, not_null=False, boolean=False, many=False):
    if len(self.terms) == 0:
        return self.default.to_python()
    elif self.default is NULL:
        return MultiOp.operators[self.op][0].join("(" + t.to_python() + ")" for t in self.terms)
    else:
        return "coalesce(" + MultiOp.operators[self.op][0].join("(" + t.to_python() + ")" for t in self.terms) + ", " + self.default.to_python() + ")"

@extend(RegExpOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "re.match(" + quote(json2value(self.pattern.json) + "$") + ", " + self.var.to_python() + ")"


@extend(CoalesceOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "coalesce(" + (",".join(t.to_python() for t in self.terms)) + ")"


@extend(MissingOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return self.expr.to_python() + " == None"


@extend(ExistsOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return self.field.to_python() + " != None"


@extend(PrefixOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.field.to_python() + ").startswith(" + self.prefix.to_python() + ")"


@extend(SuffixOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.field.to_python() + ").endswith(" + self.suffix.to_python() + ")"


@extend(ConcatOp)
def to_python(self, not_null=False, boolean=False, many=False):
    v = self.value.to_python()
    l = self.length.to_python()
    return "None if " + v + " == None or " + l + " == None else " + v + "[0:max(0, " + l + ")]"


@extend(NotLeftOp)
def to_python(self, not_null=False, boolean=False, many=False):
    v = self.value.to_python()
    l = self.length.to_python()
    return "None if " + v + " == None or " + l + " == None else " + v + "[max(0, " + l + "):]"


@extend(RightOp)
def to_python(self, not_null=False, boolean=False, many=False):
    v = self.value.to_python()
    l = self.length.to_python()
    return "None if " + v + " == None or " + l + " == None else " + v + "[max(0, len(" + v + ")-(" + l + ")):]"


@extend(NotRightOp)
def to_python(self, not_null=False, boolean=False, many=False):
    v = self.value.to_python()
    l = self.length.to_python()
    return "None if " + v + " == None or " + l + " == None else " + v + "[0:max(0, len(" + v + ")-(" + l + "))]"


@extend(SplitOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.value.to_python() + ").split(" + self.find.to_python() + ")"

@extend(FindOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "((" + quote(self.substring) + " in " + self.var.to_python() + ") if " + self.var.to_python() + "!=None else False)"


@extend(BetweenOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return self.value.to_python() + " in " + self.superset.to_python(many=True)


@extend(RangeOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.then.to_python(not_null=not_null) + ") if (" + self.when.to_python(
        boolean=True) + ") else (" + self.els_.to_python(not_null=not_null) + ")"


@extend(CaseOp)
def to_python(self, not_null=False, boolean=False, many=False):
    acc = self.whens[-1].to_python()
    for w in reversed(self.whens[0:-1]):
        acc = "(" + w.then.to_python() + ") if (" + w.when.to_python(boolean=True) + ") else (" + acc + ")"
    return acc

@extend(WhenOp)
def to_python(self, not_null=False, boolean=False, many=False):
    return "(" + self.then.to_python() + ") if (" + self.when.to_python(boolean=True) + ") else (" + self.els_.to_python() + ")"

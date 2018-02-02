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

from collections import Mapping
from copy import copy

from mo_dots import set_default, wrap, coalesce, Data, listwrap, unwraplist
from mo_logs import Log
from mo_math import Math
from mo_times.dates import Date

from jx_base.dimensions import Dimension
from jx_base.queries import is_variable_name
from jx_python.namespace import Namespace, convert_list
from jx_base.query import QueryOp


class Rename(Namespace):

    def __init__(self, dimensions, source):
        """
        EXPECTING A LIST OF {"name":name, "value":value} OBJECTS TO PERFORM A MAPPING
        """
        dimensions = wrap(dimensions)
        if isinstance(dimensions, Mapping) and dimensions.name == None:
            # CONVERT TO A REAL DIMENSION DEFINITION
            dimensions = {"name": ".", "type": "set", "edges":[{"name": k, "field": v} for k, v in dimensions.items()]}

        self.dimensions = Dimension(dimensions, None, source)

    def convert(self, expr):
        """
        EXPAND INSTANCES OF name TO value
        """
        if expr is True or expr == None or expr is False:
            return expr
        elif Math.is_number(expr):
            return expr
        elif expr == ".":
            return "."
        elif is_variable_name(expr):
            return coalesce(self.dimensions[expr], expr)
        elif isinstance(expr, text_type):
            Log.error("{{name|quote}} is not a valid variable name", name=expr)
        elif isinstance(expr, Date):
            return expr
        elif isinstance(expr, QueryOp):
            return self._convert_query(expr)
        elif isinstance(expr, Mapping):
            if expr["from"]:
                return self._convert_query(expr)
            elif len(expr) >= 2:
                #ASSUME WE HAVE A NAMED STRUCTURE, NOT AN EXPRESSION
                return wrap({name: self.convert(value) for name, value in expr.leaves()})
            else:
                # ASSUME SINGLE-CLAUSE EXPRESSION
                k, v = expr.items()[0]
                return converter_map.get(k, self._convert_bop)(self, k, v)
        elif isinstance(expr, (list, set, tuple)):
            return wrap([self.convert(value) for value in expr])
        else:
            return expr

    def _convert_query(self, query):
        output = QueryOp(None)
        output.select = self._convert_clause(query.select)
        output.where = self.convert(query.where)
        output.frum = self._convert_from(query.frum)
        output.edges = convert_list(self._convert_edge, query.edges)
        output.having = convert_list(self._convert_having, query.having)
        output.window = convert_list(self._convert_window, query.window)
        output.sort = self._convert_clause(query.sort)
        output.format = query.format

        return output




    def _convert_bop(self, op, term):
        if isinstance(term, list):
            return {op: map(self.convert, term)}

        return {op: {self.convert(var): val for var, val in term.items()}}

    def _convert_many(self, k, v):
        return {k: map(self.convert, v)}

    def _convert_from(self, frum):
        if isinstance(frum, Mapping):
            return Data(name=self.convert(frum.name))
        else:
            return self.convert(frum)

    def _convert_edge(self, edge):
        dim = self.dimensions[edge.value]
        if not dim:
            return edge

        if len(listwrap(dim.fields)) == 1:
            #TODO: CHECK IF EDGE DOMAIN AND DIMENSION DOMAIN CONFLICT
            new_edge = set_default({"value": unwraplist(dim.fields)}, edge)
            return new_edge
            new_edge.domain = dim.getDomain()

        edge = copy(edge)
        edge.value = None
        edge.domain = dim.getDomain()
        return edge

    def _convert_clause(self, clause):
        """
        JSON QUERY EXPRESSIONS HAVE MANY CLAUSES WITH SIMILAR COLUMN DELCARATIONS
        """
        clause = wrap(clause)

        if clause == None:
            return None
        elif isinstance(clause, Mapping):
            return set_default({"value": self.convert(clause.value)}, clause)
        else:
            return [set_default({"value": self.convert(c.value)}, c) for c in clause]

converter_map = {
    "and": Rename._convert_many,
    "or": Rename._convert_many,
    "not": Rename.convert,
    "missing": Rename.convert,
    "exists": Rename.convert
}


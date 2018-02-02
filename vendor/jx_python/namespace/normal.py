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

from mo_dots import Data
from mo_dots import FlatList
from mo_dots import coalesce, Null
from mo_dots import wrap, listwrap
from mo_logs import Log
from mo_math import Math

from jx_base.dimensions import Dimension
from jx_base.domains import Domain
from jx_python.containers import Container
from jx_python.expressions import TRUE
from jx_python.namespace import Namespace, convert_list
from jx_base.query import QueryOp, get_all_vars

DEFAULT_LIMIT = 10


class Normal(Namespace):
    """
    UNREMARKABLE NAMESPACE, SIMPLY FOR CONVERTING QUERY TO NORMAL FORM
    """

    def convert(self, expr):
        if isinstance(expr, Mapping) and expr["from"]:
            return self._convert_query(expr)
        return expr


    def _convert_query(self, query):
        # if not isinstance(query["from"], Container):
        #     Log.error('Expecting from clause to be a Container')
        query = wrap(query)

        output = QueryOp("from", None)
        output["from"] = self._convert_from(query["from"])

        output.format = query.format

        if query.select:
            output.select = convert_list(self._convert_select, query.select)
        else:
            if query.edges or query.groupby:
                output.select = {"name": "count", "value": ".", "aggregate": "count", "default": 0}
            else:
                output.select = {"name": "__all__", "value": "*", "aggregate": "none"}

        if query.groupby and query.edges:
            Log.error("You can not use both the `groupby` and `edges` clauses in the same query!")
        elif query.edges:
            output.edges = convert_list(self._convert_edge, query.edges)
            output.groupby = None
        elif query.groupby:
            output.edges = None
            output.groupby = convert_list(self._convert_group, query.groupby)
        else:
            output.edges = []
            output.groupby = None

        output.where = self.convert(query.where)
        output.window = convert_list(self._convert_window, query.window)
        output.sort = self._convert_sort(query.sort)

        output.limit = coalesce(query.limit, DEFAULT_LIMIT)
        if not Math.is_integer(output.limit) or output.limit < 0:
            Log.error("Expecting limit >= 0")

        output.isLean = query.isLean

        # DEPTH ANALYSIS - LOOK FOR COLUMN REFERENCES THAT MAY BE DEEPER THAN
        # THE from SOURCE IS.
        vars = get_all_vars(output, exclude_where=True)  # WE WILL EXCLUDE where VARIABLES
        for c in query.columns:
            if c.name in vars and len(c.nested_path) != 1:
                Log.error("This query, with variable {{var_name}} is too deep", var_name=c.name)

        output.having = convert_list(self._convert_having, query.having)

        return output

    def _convert_from(self, frum):
        if isinstance(frum, text_type):
            return Data(name=frum)
        elif isinstance(frum, (Container, QueryOp)):
            return frum
        else:
            Log.error("Expecting from clause to be a name, or a container")

    def _convert_select(self, select):
        if isinstance(select, text_type):
            return Data(
                name=select.rstrip("."),  # TRAILING DOT INDICATES THE VALUE, BUT IS INVALID FOR THE NAME
                value=select,
                aggregate="none"
            )
        else:
            select = wrap(select)
            output = copy(select)
            if not select.value or isinstance(select.value, text_type):
                if select.value == ".":
                    output.name = coalesce(select.name, select.aggregate)
                else:
                    output.name = coalesce(select.name, select.value, select.aggregate)
            elif not output.name:
                Log.error("Must give name to each column in select clause")

            if not output.name:
                Log.error("expecting select to have a name: {{select}}",  select=select)

            output.aggregate = coalesce(canonical_aggregates.get(select.aggregate), select.aggregate, "none")
            return output

    def _convert_edge(self, edge):
        if isinstance(edge, text_type):
            return Data(
                name=edge,
                value=edge,
                domain=self._convert_domain()
            )
        else:
            edge = wrap(edge)
            if not edge.name and not isinstance(edge.value, text_type):
                Log.error("You must name compound edges: {{edge}}",  edge= edge)

            if isinstance(edge.value, (Mapping, list)) and not edge.domain:
                # COMPLEX EDGE IS SHORT HAND
                domain =self._convert_domain()
                domain.dimension = Data(fields=edge.value)

                return Data(
                    name=edge.name,
                    allowNulls=False if edge.allowNulls is False else True,
                    domain=domain
                )

            domain = self._convert_domain(edge.domain)
            return Data(
                name=coalesce(edge.name, edge.value),
                value=edge.value,
                range=edge.range,
                allowNulls=False if edge.allowNulls is False else True,
                domain=domain
            )

    def _convert_group(self, column):
        if isinstance(column, text_type):
            return wrap({
                "name": column,
                "value": column,
                "domain": {"type": "default"}
            })
        else:
            column = wrap(column)
            if (column.domain and column.domain.type != "default") or column.allowNulls != None:
                Log.error("groupby does not accept complicated domains")

            if not column.name and not isinstance(column.value, text_type):
                Log.error("You must name compound edges: {{edge}}",  edge= column)

            return wrap({
                "name": coalesce(column.name, column.value),
                "value": column.value,
                "domain": {"type": "default"}
            })


    def _convert_domain(self, domain=None):
        if not domain:
            return Domain(type="default")
        elif isinstance(domain, Dimension):
            return domain.getDomain()
        elif isinstance(domain, Domain):
            return domain

        if not domain.name:
            domain = domain.copy()
            domain.name = domain.type

        if not isinstance(domain.partitions, list):
            domain.partitions = list(domain.partitions)

        return Domain(**domain)

    def _convert_range(self, range):
        if range == None:
            return None

        return Data(
            min=range.min,
            max=range.max
        )

    def _convert_where(self, where):
        if where == None:
            return TRUE
        return where


    def _convert_window(self, window):
        return Data(
            name=coalesce(window.name, window.value),
            value=window.value,
            edges=[self._convert_edge(e) for e in listwrap(window.edges)],
            sort=self._convert_sort(window.sort),
            aggregate=window.aggregate,
            range=self._convert_range(window.range),
            where=self._convert_where(window.where)
        )


    def _convert_sort(self, sort):
        return normalize_sort(sort)


def normalize_sort(sort=None):
    """
    CONVERT SORT PARAMETERS TO A NORMAL FORM SO EASIER TO USE
    """

    if not sort:
        return Null

    output = FlatList()
    for s in listwrap(sort):
        if isinstance(s, text_type) or Math.is_integer(s):
            output.append({"value": s, "sort": 1})
        elif not s.field and not s.value and s.sort==None:
            #ASSUME {name: sort} FORM
            for n, v in s.items():
                output.append({"value": n, "sort": sort_direction[v]})
        else:
            output.append({"value": coalesce(s.field, s.value), "sort": coalesce(sort_direction[s.sort], 1)})
    return wrap(output)


sort_direction = {
    "asc": 1,
    "desc": -1,
    "none": 0,
    1: 1,
    0: 0,
    -1: -1,
    None: 1,
    Null: 1
}

canonical_aggregates = {
    "none": "none",
    "one": "one",
    "count": "count",
    "sum": "sum",
    "add": "sum",
    "mean": "average",
    "average": "average",
    "avg": "average",
    "min": "minimum",
    "minimum": "minimum",
    "max": "maximum",
    "maximum": "minimum",
    "X2": "sum_of_squares",
    "std": "std",
    "stddev": "std",
    "std_deviation": "std",
    "var": "variance",
    "variance": "variance",
    "stats": "stats"
}


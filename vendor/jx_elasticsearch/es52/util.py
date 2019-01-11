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

from jx_base.expressions import Variable
from jx_base.language import is_op
from mo_dots import wrap
from mo_future import is_text
from mo_json import BOOLEAN, IS_NULL, NUMBER, OBJECT, STRING
from mo_logs import Log
from pyLibrary.convert import value2boolean


def es_query_template(path):
    """
    RETURN TEMPLATE AND PATH-TO-FILTER AS A 2-TUPLE
    :param path: THE NESTED PATH (NOT INCLUDING TABLE NAME)
    :return: (es_query, es_filters) TUPLE
    """

    if not is_text(path):
        Log.error("expecting path to be a string")

    if path != ".":
        f0 = {}
        f1 = {}
        output = wrap({
            "query": es_and([
                f0,
                {"nested": {
                    "path": path,
                    "query": f1,
                    "inner_hits": {"size": 100000}
                }}
            ]),
            "from": 0,
            "size": 0,
            "sort": []
        })
        return output, wrap([f0, f1])
    else:
        f0 = {}
        output = wrap({
            "query": es_and([f0]),
            "from": 0,
            "size": 0,
            "sort": []
        })
        return output, wrap([f0])


def jx_sort_to_es_sort(sort, schema):
    if not sort:
        return []

    output = []
    for s in sort:
        if is_op(s.value, Variable):
            cols = schema.leaves(s.value.var)
            if s.sort == -1:
                types = OBJECT, STRING, NUMBER, BOOLEAN
            else:
                types = BOOLEAN, NUMBER, STRING, OBJECT

            for type in types:
                for c in cols:
                    if c.jx_type is type:
                        if s.sort == -1:
                            output.append({c.es_column: "desc"})
                        else:
                            output.append(c.es_column)
        else:
            from mo_logs import Log

            Log.error("do not know how to handle")
    return output


# FOR ELASTICSEARCH aggs
aggregates = {
    "none": "none",
    "one": "count",
    "cardinality": "cardinality",
    "sum": "sum",
    "add": "sum",
    "count": "value_count",
    "count_values": "count_values",
    "maximum": "max",
    "minimum": "min",
    "max": "max",
    "min": "min",
    "mean": "avg",
    "average": "avg",
    "avg": "avg",
    "median": "median",
    "percentile": "percentile",
    "N": "count",
    "s0": "count",
    "s1": "sum",
    "s2": "sum_of_squares",
    "std": "std_deviation",
    "stddev": "std_deviation",
    "union": "union",
    "var": "variance",
    "variance": "variance",
    "stats": "stats"
}

NON_STATISTICAL_AGGS = {"none", "one"}

def es_and(terms):
    return wrap({"bool": {"filter": terms}})


def es_or(terms):
    return wrap({"bool": {"should": terms}})


def es_not(term):
    return wrap({"bool": {"must_not": term}})


def es_script(term):
    return wrap({"script": {"lang": "painless", "source": term}})


def es_missing(term):
    return {"bool": {"must_not": {"exists": {"field": term}}}}


def es_exists(term):
    return {"exists": {"field": term}}


MATCH_ALL = wrap({"match_all": {}})
MATCH_NONE = es_not({"match_all": {}})


pull_functions = {
    IS_NULL: lambda x: None,
    STRING: lambda x: x,
    NUMBER: lambda x: float(x) if x !=None else None,
    BOOLEAN: value2boolean
}

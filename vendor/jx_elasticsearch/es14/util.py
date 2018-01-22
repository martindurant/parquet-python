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

from jx_base import STRING, BOOLEAN, NUMBER, OBJECT
from jx_elasticsearch.es14.expressions import Variable
from mo_dots import wrap


def es_query_template(path):
    """
    RETURN TEMPLATE AND PATH-TO-FILTER AS A 2-TUPLE
    :param path: THE NESTED PATH (NOT INCLUDING TABLE NAME)
    :return:
    """

    if path != ".":
        f0 = {}
        f1 = {}
        output = wrap({
            "query": {"filtered": {"filter": {"and":[
                f0,
                {"nested": {
                    "path": path,
                    "filter": f1,
                    "inner_hits": {"size": 100000}
                }}
            ]}}},
            "from": 0,
            "size": 0,
            "sort": []
        })
        return output, wrap([f0, f1])
    else:
        f0 = {}
        output = wrap({
            "query": {"filtered": {"filter": f0}},
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
        if isinstance(s.value, Variable):
            cols = schema.leaves(s.value.var)
            if s.sort == -1:
                types = OBJECT, STRING, NUMBER, BOOLEAN
            else:
                types = BOOLEAN, NUMBER, STRING, OBJECT

            for type in types:
                for c in cols:
                    if c.type == type:
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


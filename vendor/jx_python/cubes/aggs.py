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

from jx_python import windows
from mo_dots import listwrap
from mo_logs import Log

from jx_base.domains import SimpleSetDomain, DefaultDomain
from jx_python.expressions import jx_expression_to_function
from mo_collections.matrix import Matrix


def cube_aggs(frum, query):
    select = listwrap(query.select)

    #MATCH EDGES IN QUERY TO ONES IN frum
    for e in query.edges:
        for fs in frum.select:
            if fs.name == e.value:
                Log.error("Not implemented yet")
        if isinstance(e.domain, DefaultDomain):
            # DEFAULT DOMAINS CAN EASILY BE LOOKED UP FROM frum
            for fe in frum.edges:
                if fe.name == e.value:
                    e.domain = SimpleSetDomain(**fe.domain.__data__())
                    e.value = e.value + "." + fe.domain.key
                    break
        else:
            for fe in frum.edges:
                if fe.name == e.value:
                    e.value = e.value + "." + fe.domain.key
                    break


    result = {
        s.name: Matrix(
            dims=[len(e.domain.partitions) + (1 if e.allowNulls else 0) for e in query.edges],
            zeros=s.default
        )
        for s in select
    }
    where = jx_expression_to_function(query.where)
    for d in filter(where, frum.values()):
        coord = []  # LIST OF MATCHING COORDINATE FAMILIES, USUALLY ONLY ONE PER FAMILY BUT JOINS WITH EDGES CAN CAUSE MORE
        for e in query.edges:
            matches = get_matches(e, d)
            coord.append(matches)
            if len(matches) == 1 and d[e.name] == None:
                d[e.name] = e.domain.partitions[matches[0]]

        for s in select:
            mat = result[s.name]
            agg = s.aggregate
            var = s.value
            expr = jx_expression_to_function(var)
            val = expr(d)
            if agg == "count":
                if var == "." or var == None:
                    for c in itertools.product(*coord):
                        mat[c] += 1
                    continue

                if val != None:
                    for c in itertools.product(*coord):
                        mat[c] += 1
            else:
                for c in itertools.product(*coord):
                    acc = mat[c]
                    if acc == None:
                        acc = windows.name2accumulator.get(agg)
                        if acc == None:
                            Log.error("select aggregate {{agg}} is not recognized",  agg= agg)
                        acc = acc(**s)
                        mat[c] = acc
                    acc.add(val)

    for s in select:
        if s.aggregate == "count":
            continue
        m = result[s.name]
        for c, var in m.items():
            if var != None:
                m[c] = var.end()

    from jx_python.containers.cube import Cube

    return Cube(select, query.edges, result)


def get_matches(e, d):
    if e.value:
        return [e.domain.getIndexByKey(d[e.value])]
    elif e.range:
        output = []
        mi, ma = d[e.range.min], d[e.range.max]
        var = e.domain.key
        for p in e.domain.partitions:
            if mi <= p[var] < ma:
                output.append(p.dataIndex)
        if e.allowNulls and not output:
            output.append(len(e.domain.partitions))  # ENSURE THIS IS NULL
        return output

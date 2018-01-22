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

from jx_base import domains
from jx_base.expressions import TRUE, jx_expression, Variable, LeavesOp
from jx_base.queries import is_variable_name
from jx_elasticsearch import es09
from jx_elasticsearch.es09.expressions import unpack_terms
from jx_elasticsearch.es09.util import aggregates
from jx_elasticsearch.es09.util import post as es_post
from jx_python.containers.cube import Cube
from mo_collections.matrix import Matrix
from mo_dots import coalesce, split_field, Data, wrap
from mo_dots import listwrap, unwrap
from mo_dots.lists import FlatList
from mo_logs import Log
from mo_math import AND, SUM, OR


def is_fieldop(query):
    # THESE SMOOTH EDGES REQUIRE ALL DATA (SETOP)

    select = listwrap(query.select)
    if not query.edges:
        isDeep = len(split_field(query.frum.name)) > 1  # LOOKING INTO NESTED WILL REQUIRE A SCRIPT
        isSimple = AND(s.value != None and (s.value == "*" or is_variable_name(s.value)) for s in select)
        noAgg = AND(s.aggregate == "none" for s in select)

        if not isDeep and isSimple and noAgg:
            return True
    else:
        isSmooth = AND((e.domain.type in domains.ALGEBRAIC and e.domain.interval == "none") for e in query.edges)
        if isSmooth:
            return True

    return False


def es_fieldop(es, query):
    FromES = es09.util.build_es_query(query)
    select = listwrap(query.select)
    FromES.query = {
        "bool": {
            "query": {
                "match_all": {}
            },
            "filter": jx_expression(query.where).to_esfilter()
        }
    }
    FromES.size = coalesce(query.limit, 200000)
    FromES.fields = FlatList()
    for s in select.value:
        if s == "*":
            FromES.fields = None
        elif isinstance(s, list):
            FromES.fields.extend(s)
        elif isinstance(s, Mapping):
            FromES.fields.extend(s.values())
        else:
            FromES.fields.append(s)
    FromES.sort = [{s.field: "asc" if s.sort >= 0 else "desc"} for s in query.sort]

    data = es_post(es, FromES, query.limit)

    T = data.hits.hits
    matricies = {}
    for s in select:
        if s.value == "*":
            matricies[s.name] = Matrix.wrap([t._source for t in T])
        elif isinstance(s.value, Mapping):
            # for k, v in s.value.items():
            #     matricies[join_field(split_field(s.name)+[k])] = Matrix.wrap([unwrap(t.fields)[v] for t in T])
            matricies[s.name] = Matrix.wrap([{k: unwrap(t.fields).get(v, None) for k, v in s.value.items()}for t in T])
        elif isinstance(s.value, list):
            matricies[s.name] = Matrix.wrap([tuple(unwrap(t.fields).get(ss, None) for ss in s.value) for t in T])
        elif not s.value:
            matricies[s.name] = Matrix.wrap([unwrap(t.fields).get(s.value, None) for t in T])
        else:
            try:
                matricies[s.name] = Matrix.wrap([unwrap(t.fields).get(s.value, None) for t in T])
            except Exception as e:
                Log.error("", e)

    cube = Cube(query.select, query.edges, matricies, frum=query)
    cube.frum = query
    return cube


def is_setop(query):
    select = listwrap(query.select)

    if not query.edges:
        isDeep = len(split_field(query.frum.name)) > 1  # LOOKING INTO NESTED WILL REQUIRE A SCRIPT
        simpleAgg = AND([s.aggregate in ("count", "none") for s in select])   # CONVERTING esfilter DEFINED PARTS WILL REQUIRE SCRIPT

        # NO EDGES IMPLIES SIMPLER QUERIES: EITHER A SET OPERATION, OR RETURN SINGLE AGGREGATE
        if simpleAgg or isDeep:
            return True
    else:
        isSmooth = AND((e.domain.type in domains.ALGEBRAIC and e.domain.interval == "none") for e in query.edges)
        if isSmooth:
            return True

    return False


def es_setop(es, mvel, query):
    FromES = es09.util.build_es_query(query)
    select = listwrap(query.select)

    isDeep = len(split_field(query.frum.name)) > 1  # LOOKING INTO NESTED WILL REQUIRE A SCRIPT
    isComplex = OR([s.value == None and s.aggregate not in ("count", "none") for s in select])   # CONVERTING esfilter DEFINED PARTS WILL REQUIRE SCRIPT

    if not isDeep and not isComplex:
        if len(select) == 1 and isinstance(select[0].value, LeavesOp):
            FromES = wrap({
                "query": {"bool": {
                    "query": {"match_all": {}},
                    "filter": query.where.to_esfilter()
                }},
                "sort": query.sort,
                "size": 0
            })
        elif all(isinstance(v, Variable) for v in select.value):
            FromES = wrap({
                "query": {"bool": {
                    "query": {"match_all": {}},
                    "filter": query.where.to_esfilter()
                }},
                "fields": select.value,
                "sort": query.sort,
                "size": coalesce(query.limit, 200000)
            })
    elif not isDeep:
        simple_query = query.copy()
        simple_query.where = TRUE  # THE FACET FILTER IS FASTER
        FromES.facets.mvel = {
            "terms": {
                "script_field": mvel.code(simple_query),
                "size": coalesce(simple_query.limit, 200000)
            },
            "facet_filter": jx_expression(query.where).to_esfilter()
        }
    else:
        FromES.facets.mvel = {
            "terms": {
                "script_field": mvel.code(query),
                "size": coalesce(query.limit, 200000)
            },
            "facet_filter": jx_expression(query.where).to_esfilter()
        }

    data = es_post(es, FromES, query.limit)

    if len(select) == 1 and isinstance(select[0].value, LeavesOp):
        # SPECIAL CASE FOR SINGLE COUNT
        cube = wrap(data).hits.hits._source
    elif isinstance(select[0].value, Variable):
        # SPECIAL CASE FOR SINGLE TERM
        cube = wrap(data).hits.hits.fields
    else:
        data_list = unpack_terms(data.facets.mvel, select)
        if not data_list:
            cube = Cube(select, [], {s.name: Matrix.wrap([]) for s in select})
        else:
            output = zip(*data_list)
            cube = Cube(select, [], {s.name: Matrix(list=output[i]) for i, s in enumerate(select)})

    return Data(
        meta={"esquery": FromES},
        data=cube
    )


def is_deep(query):
    select = listwrap(query.select)
    if len(select) > 1:
        return False

    if aggregates[select[0].aggregate] not in ("none", "count"):
        return False

    if len(query.edges)<=1:
        return False

    isDeep = len(split_field(query["from"].name)) > 1  # LOOKING INTO NESTED WILL REQUIRE A SCRIPT
    if not isDeep:
        return False   # BETTER TO USE TERM QUERY

    return True


def es_deepop(es, mvel, query):
    FromES = es09.util.build_es_query(query)

    select = query.edges

    temp_query = query.copy()
    temp_query.select = select
    temp_query.edges = FlatList()
    FromES.facets.mvel = {
        "terms": {
            "script_field": mvel.code(temp_query),
            "size": query.limit
        },
        "facet_filter": jx_expression(query.where).to_esfilter()
    }

    data = es_post(es, FromES, query.limit)

    rows = unpack_terms(data.facets.mvel, query.edges)
    terms = zip(*rows)

    # NUMBER ALL EDGES FOR JSON EXPRESSION INDEXING
    edges = query.edges
    for f, e in enumerate(edges):
        for r in terms[f]:
            e.domain.getPartByKey(r)

        e.index = f
        for p, part in enumerate(e.domain.partitions):
            part.dataIndex = p
        e.domain.NULL.dataIndex = len(e.domain.partitions)

    # MAKE CUBE
    dims = [len(e.domain.partitions) for e in query.edges]
    output = Matrix(*dims)

    # FILL CUBE
    for r in rows:
        term_coord = [e.domain.getPartByKey(r[i]).dataIndex for i, e in enumerate(edges)]
        output[term_coord] = SUM(output[term_coord], r[-1])

    cube = Cube(query.select, query.edges, {query.select.name: output})
    cube.frum = query
    return cube

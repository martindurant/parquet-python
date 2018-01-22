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

from jx_elasticsearch.es09.util import aggregates, build_es_query, compileEdges2Term
from jx_elasticsearch.es09.util import post as es_post
from jx_python import jx
from jx_python.containers.cube import Cube
from mo_collections.matrix import Matrix
from mo_dots import coalesce
from mo_dots import wrap, listwrap
from mo_dots.lists import FlatList
from mo_math import AND


def is_terms(query):
    select = listwrap(query.select)

    isSimple = not query.select or AND(aggregates[s.aggregate] in ("none", "count") for s in select)
    if isSimple:
        return True
    return False


def es_terms(es, mvel, query):
    """
    RETURN LIST OF ALL EDGE QUERIES

    EVERY FACET IS NAMED <select.name>, <c1>, ... <cN> WHERE <ci> ARE THE ELEMENT COORDINATES
    WE TRY TO PACK DIMENSIONS INTO THE TERMS TO MINIMIZE THE CROSS-PRODUCT EXPLOSION
    """
    if len(query.edges) == 2:
        return _es_terms2(es, mvel, query)

    select = listwrap(query.select)
    FromES = build_es_query(query)
    packed_term = compileEdges2Term(mvel, query.edges, wrap([]))
    for s in select:
        FromES.facets[s.name] = {
            "terms": {
                "field": packed_term.field,
                "script_field": packed_term.expression,
                "size": coalesce(query.limit, 200000)
            },
            "facet_filter": simplify_esfilter(query.where)
        }

    term2Parts = packed_term.term2parts

    data = es_post(es, FromES, query.limit)

    # GETTING ALL PARTS WILL EXPAND THE EDGES' DOMAINS
    # BUT HOW TO UNPACK IT FROM THE term FASTER IS UNKNOWN
    for k, f in data.facets.items():
        for t in f.terms:
            term2Parts(t.term)

    # NUMBER ALL EDGES FOR jx INDEXING
    for f, e in enumerate(query.edges):
        e.index = f
        if e.domain.type in ["uid", "default"]:
            # e.domain.partitions = jx.sort(e.domain.partitions, "value")
            for p, part in enumerate(e.domain.partitions):
                part.dataIndex = p
            e.domain.NULL.dataIndex = len(e.domain.partitions)

    # MAKE CUBE
    output = {}
    dims = [len(e.domain.partitions) + (1 if e.allowNulls else 0) for e in query.edges]
    for s in select:
        output[s.name] = Matrix(*dims)

    # FILL CUBE
    # EXPECTING ONLY SELECT CLAUSE FACETS
    for facetName, facet in data.facets.items():
        for term in facet.terms:
            term_coord = term2Parts(term.term).dataIndex
            for s in select:
                try:
                    output[s.name][term_coord] = term[aggregates[s.aggregate]]
                except Exception as e:
                    # USUALLY CAUSED BY output[s.name] NOT BEING BIG ENOUGH TO HANDLE NULL COUNTS
                    pass
    cube = Cube(query.select, query.edges, output)
    cube.query = query
    return cube


def _es_terms2(es, mvel, query):
    """
    WE ASSUME THERE ARE JUST TWO EDGES, AND EACH HAS A SIMPLE value
    """

    # REQUEST VALUES IN FIRST DIMENSION
    q1 = query.copy()
    q1.edges = query.edges[0:1:]
    values1 = es_terms(es, mvel, q1).edges[0].domain.partitions.value

    select = listwrap(query.select)
    FromES = build_es_query(query)
    for s in select:
        for i, v in enumerate(values1):
            FromES.facets[s.name + "," + str(i)] = {
                "terms": {
                    "field": query.edges[1].value,
                    "size": coalesce(query.limit, 200000)
                },
                "facet_filter": simplify_esfilter({"and": [
                    query.where,
                    {"term": {query.edges[0].value: v}}
                ]})
            }

    data = es_post(es, FromES, query.limit)

    # UNION ALL TERMS FROM SECOND DIMENSION
    values2 = set()
    for k, f in data.facets.items():
        values2.update(f.terms.term)
    values2 = jx.sort(values2)
    term2index = {v: i for i, v in enumerate(values2)}
    query.edges[1].domain.partitions = FlatList([{"name": v, "value": v} for v in values2])

    # MAKE CUBE
    output = {}
    dims = [len(values1), len(values2)]
    for s in select:
        output[s.name] = Matrix(*dims)

    # FILL CUBE
    # EXPECTING ONLY SELECT CLAUSE FACETS
    for facetName, facet in data.facets.items():
        coord = facetName.split(",")
        s = [s for s in select if s.name == coord[0]][0]
        i1 = int(coord[1])
        for term in facet.terms:
            i2 = term2index[term.term]
            output[s.name][(i1, i2)] = term[aggregates[s.aggregate]]

    cube = Cube(query.select, query.edges, output)
    cube.query = query
    return cube


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

from jx_base.queries import is_variable_name
from jx_elasticsearch.es09.expressions import UID
from jx_elasticsearch.es09.util import aggregates, build_es_query, compileEdges2Term
from jx_python import domains
from jx_python import es09
from jx_python.containers.cube import Cube
from jx_python.expressions import simplify_esfilter
from mo_collections.matrix import Matrix
from mo_dots import literal_field, coalesce
from mo_dots import wrap, listwrap
from mo_dots.lists import FlatList
from mo_logs import Log
from mo_math import COUNT, PRODUCT


def is_terms_stats(query):
    # ONLY ALLOWED ONE UNKNOWN DOMAIN
    num_unknown = COUNT(1 for e in query.edges if e.domain.type not in domains.KNOWN)

    if num_unknown <= 1:
        if query.sort:
            Log.error("terms_stats can not be sorted")

        return True
    return False


def es_terms_stats(esq, mvel, query):
    select = listwrap(query.select)
    facetEdges = []    # EDGES THAT WILL REQUIRE A FACET FOR EACH PART
    termsEdges = FlatList()
    specialEdge = None
    special_index = -1

    # A SPECIAL EDGE IS ONE THAT HAS AN UNDEFINED NUMBER OF PARTITIONS AT QUERY TIME
    # FIND THE specialEdge, IF ONE
    for f, tedge in enumerate(query.edges):
        if tedge.domain.type in domains.KNOWN:
            for p, part in enumerate(tedge.domain.partitions):
                part.dataIndex = p

            # FACETS ARE ONLY REQUIRED IF SQL JOIN ON DOMAIN IS REQUIRED (RANGE QUERY)
            # OR IF WE ARE NOT SIMPLY COUNTING
            # OR IF NO SCRIPTING IS ALLOWED (SOME OTHER CODE IS RESPONSIBLE FOR SETTING isFacet)
            # OR IF WE JUST WANT TO FORCE IT :)
            # OF COURSE THE default EDGE IS NOT EXPLICIT, SO MUST BE A TERM

            facetEdges.append(tedge)
        else:
            if specialEdge:
                Log.error("There is more than one open-ended edge: self can not be handled")
            specialEdge = tedge
            special_index = f
            termsEdges.append(tedge)

    if not specialEdge:
        # WE SERIOUSLY WANT A SPECIAL EDGE, OTHERWISE WE WILL HAVE TOO MANY FACETS
        # THE BIGGEST EDGE MAY BE COLLAPSED TO A TERM, MAYBE?
        num_parts = 0
        special_index = -1
        for i, e in enumerate(facetEdges):
            l = len(e.domain.partitions)
            if ((e.value and is_variable_name(e.value)) or len(e.domain.dimension.fields) == 1) and l > num_parts:
                num_parts = l
                specialEdge = e
                special_index = i

        facetEdges.pop(special_index)
        termsEdges.append(specialEdge)

    total_facets = PRODUCT(len(f.domain.partitions) for f in facetEdges)*len(select)
    if total_facets > 100:
        # WE GOT A PROBLEM, LETS COUNT THE SIZE OF REALITY:
        counts = esq.query({
            "from": query.frum,
            "select": {"aggregate": "count"},
            "edges": facetEdges,
            "where": query.where,
            "limit": query.limit
        })

        esFacets = []

        def add_facet(value, parts, cube):
            if value:
                esFacets.append(parts)

        counts["count"].forall(add_facet)

        Log.note("{{theory_count}} theoretical combinations, {{real_count}} actual combos found",  real_count= len(esFacets),  theory_count=total_facets)

        if not esFacets:
            # MAKE EMPTY CUBE
            matricies = {}
            dims = [len(e.domain.partitions) + (1 if e.allowNulls else 0) for e in query.edges]
            for s in select:
                matricies[s.name] = Matrix(*dims)
            cube = Cube(query.select, query.edges, matricies)
            cube.frum = query
            return cube

    else:
        # GENERATE ALL COMBOS
        esFacets = getAllEdges(facetEdges)

    calcTerm = compileEdges2Term(mvel, termsEdges, FlatList())
    term2parts = calcTerm.term2parts

    if len(esFacets) * len(select) > 1000:
        Log.error("not implemented yet")  # WE HAVE SOME SERIOUS PERMUTATIONS, WE MUST ISSUE MULTIPLE QUERIES
        pass

    FromES = build_es_query(query)

    for s in select:
        for parts in esFacets:
            condition = FlatList()
            constants = FlatList()
            name = [literal_field(s.name)]
            for f, fedge in enumerate(facetEdges):
                name.append(str(parts[f].dataIndex))
                condition.append(buildCondition(mvel, fedge, parts[f]))
                constants.append({"name": fedge.domain.name, "value": parts[f]})
            condition.append(query.where)
            name = ",".join(name)

            FromES.facets[name] = {
                "terms_stats": {
                    "key_field": calcTerm.field,
                    "value_field": s.value if is_variable_name(s.value) else None,
                    "value_script": mvel.compile_expression(s.value) if not is_variable_name(s.value) else None,
                    "size": coalesce(query.limit, 200000)
                }
            }
            if condition:
                FromES.facets[name].facet_filter = simplify_esfilter({"and": condition})

    data = es_post(esq.es, FromES, query.limit)

    if specialEdge.domain.type not in domains.KNOWN:
        # WE BUILD THE PARTS BASED ON THE RESULTS WE RECEIVED
        partitions = FlatList()
        map = {}
        for facetName, parts in data.facets.items():
            for stats in parts.terms:
                if not map[stats]:
                    part = {"value": stats, "name": stats}
                    partitions.append(part)
                    map[stats] = part

        partitions.sort(specialEdge.domain.compare)
        for p, part in enumerate(partitions):
            part.dataIndex = p

        specialEdge.domain.map = map
        specialEdge.domain.partitions = partitions

    # MAKE CUBE
    matricies = {}
    dims = [len(e.domain.partitions) + (1 if e.allowNulls else 0) for e in query.edges]
    for s in select:
        matricies[s.name] = Matrix(*dims)

    name2agg = {s.name: aggregates[s.aggregate] for s in select}

    # FILL CUBE
    for edgeName, parts in data.facets.items():
        temp = edgeName.split(",")
        pre_coord = tuple(int(c) for c in temp[1:])
        sname = temp[0]

        for stats in parts.terms:
            if specialEdge:
                special = term2parts(stats.term)[0]
                coord = pre_coord[:special_index]+(special.dataIndex, )+pre_coord[special_index:]
            else:
                coord = pre_coord
            matricies[sname][coord] = stats[name2agg[sname]]

    cube = Cube(query.select, query.edges, matricies)
    cube.frum = query
    return cube


def register_script_field(FromES, code):
    if not FromES.script_fields:
        FromES.script_fields = {}

    # IF CODE IS IDENTICAL, THEN USE THE EXISTING SCRIPT
    for n, c in FromES.script_fields.items():
        if c.script == code:
            return n

    name = "script" + UID()
    FromES.script_fields[name].script = code
    return name


def getAllEdges(facetEdges):
    if not facetEdges:
        return [()]
    return _getAllEdges(facetEdges, 0)


def _getAllEdges(facetEdges, edgeDepth):
    """
    RETURN ALL PARTITION COMBINATIONS:  A LIST OF ORDERED TUPLES
    """
    if edgeDepth == len(facetEdges):
        return [()]
    edge = facetEdges[edgeDepth]

    deeper = _getAllEdges(facetEdges, edgeDepth + 1)

    output = FlatList()
    partitions = edge.domain.partitions
    for part in partitions:
        for deep in deeper:
            output.append((part,) + deep)
    return output


def buildCondition(mvel, edge, partition):
    """
    RETURN AN ES FILTER OBJECT
    """
    output = {}

    if edge.domain.isFacet:
        # MUST USE THIS' esFacet
        condition = wrap(coalesce(partition.where, {"and": []}))

        if partition.min and partition.max and is_variable_name(edge.value):
            condition["and"].append({
                "range": {edge.value: {"gte": partition.min, "lt": partition.max}}
            })

        # ES WILL FREAK OUT IF WE SEND {"not":{"and":x}} (OR SOMETHING LIKE THAT)
        return simplify_esfilter(condition)
    elif edge.range:
        # THESE REALLY NEED FACETS TO PERFORM THE JOIN-TO-DOMAIN
        # USE MVEL CODE
        if edge.domain.type in domains.ALGEBRAIC:
            output = {"and": []}

            if edge.range.mode and edge.range.mode == "inclusive":
                # IF THE range AND THE partition OVERLAP, THEN MATCH IS MADE
                if is_variable_name(edge.range.min):
                    output["and"].append({"range": {edge.range.min: {"lt": es09.expressions.value2value(partition.max)}}})
                else:
                    # WHOA!! SUPER SLOW!!
                    output["and"].append({"script": {"script": mvel.compile_expression(
                        edge.range.min + " < " + es09.expressions.value2MVEL(partition.max)
                    )}})

                if is_variable_name(edge.range.max):
                    output["and"].append({"or": [
                        {"missing": {"field": edge.range.max}},
                        {"range": {edge.range.max, {"gt": es09.expressions.value2value(partition.min)}}}
                    ]})
                else:
                    # WHOA!! SUPER SLOW!!
                    output["and"].append({"script": {"script": mvel.compile_expression(
                        edge.range.max + " > " + es09.expressions.value2MVEL(partition.min))}})

            else:
                # SNAPSHOT - IF range INCLUDES partition.min, THEN MATCH IS MADE
                if is_variable_name(edge.range.min):
                    output["and"].append({"range": {edge.range.min: {"lte": es09.expressions.value2value(partition.min)}}})
                else:
                    # WHOA!! SUPER SLOW!!
                    output["and"].append({"script": {"script": mvel.compile_expression(
                        edge.range.min + "<=" + es09.expressions.value2MVEL(partition.min)
                    )}})

                if is_variable_name(edge.range.max):
                    output["and"].append({"or": [
                        {"missing": {"field": edge.range.max}},
                        {"range": {edge.range.max, {"gte": es09.expressions.value2value(partition.min)}}}
                    ]})
                else:
                    # WHOA!! SUPER SLOW!!
                    output["and"].append({"script": {"script": mvel.compile_expression(
                        es09.expressions.value2MVEL(partition.min) + " <= " + edge.range.max
                    )}})
            return output
        else:
            Log.error("Do not know how to handle range query on non-continuous domain")

    elif not edge.value:
        # MUST USE THIS' esFacet, AND NOT(ALL THOSE ABOVE)
        return partition.esfilter
    elif is_variable_name(edge.value):
        # USE FAST ES SYNTAX
        if edge.domain.type in domains.ALGEBRAIC:
            output.range = {}
            output.range[edge.value] = {"gte": es09.expressions.value2query(partition.min), "lt": es09.expressions.value2query(partition.max)}
        elif edge.domain.type == "set":
            if partition.value:
                if partition.value != edge.domain.getKey(partition):
                    Log.error("please ensure the key attribute of the domain matches the value attribute of all partitions, if only because we are now using the former")
                    # DEFAULT TO USING THE .value ATTRIBUTE, IF ONLY BECAUSE OF LEGACY REASONS
                output.term = {edge.value: partition.value}
            else:
                output.term = {edge.value: edge.domain.getKey(partition)}

        elif edge.domain.type == "default":
            output.term = dict()
            output.term[edge.value] = partition.value
        else:
            Log.error("Edge \"" + edge.name + "\" is not supported")

        return output
    else:
        # USE MVEL CODE
        if edge.domain.type in domains.ALGEBRAIC:
            output.script = {"script": edge.value + ">=" + es09.expressions.value2MVEL(partition.min) + " and " + edge.value + "<" + es09.expressions.value2MVEL(partition.max)}
        else:
            output.script = {"script": "( " + edge.value + " ) ==" + es09.expressions.value2MVEL(partition.value)}

        code = es09.expressions.addFunctions(output.script.script)
        output.script.script = code.head + code.body
        return output


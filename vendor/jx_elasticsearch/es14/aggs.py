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

from mo_future import text_type

from jx_base.domains import SetDomain
from jx_base.expressions import TupleOp, NULL
from jx_base.query import DEFAULT_LIMIT, MAX_LIMIT
from jx_elasticsearch.es09.util import post as es_post
from jx_elasticsearch.es14.decoders import DefaultDecoder, AggsDecoder, ObjectDecoder
from jx_elasticsearch.es14.decoders import DimFieldListDecoder
from jx_elasticsearch.es14.expressions import split_expression_by_depth, AndOp, Variable, NullOp
from jx_elasticsearch.es14.setop import get_pull_stats
from jx_elasticsearch.es14.util import aggregates
from jx_python import jx
from jx_python.expressions import jx_expression_to_function
from mo_dots import listwrap, Data, wrap, literal_field, set_default, coalesce, Null, split_field, FlatList, unwrap, unwraplist
from mo_json.typed_encoder import encode_property
from mo_logs import Log
from mo_logs.strings import quote
from mo_math import Math, MAX, UNION
from mo_times.timer import Timer


def is_aggsop(es, query):
    es.cluster.get_metadata()
    if query.edges or query.groupby or any(a != None and a != "none" for a in listwrap(query.select).aggregate):
        return True
    return False


def get_decoders_by_depth(query):
    """
    RETURN A LIST OF DECODER ARRAYS, ONE ARRAY FOR EACH NESTED DEPTH
    """
    schema = query.frum.schema
    output = FlatList()

    if query.edges:
        if query.sort and query.format != "cube":
            # REORDER EDGES/GROUPBY TO MATCH THE SORT
            query.edges = sort_edges(query, "edges")
    elif query.groupby:
        if query.sort and query.format != "cube":
            query.groupby = sort_edges(query, "groupby")

    for edge in wrap(coalesce(query.edges, query.groupby, [])):
        limit = coalesce(edge.domain.limit, query.limit, DEFAULT_LIMIT)
        if edge.value != None and not isinstance(edge.value, NullOp):
            edge = edge.copy()
            vars_ = edge.value.vars()
            for v in vars_:
                if not schema.leaves(v, meta=True):
                    Log.error("{{var}} does not exist in schema", var=v)
        elif edge.range:
            vars_ = edge.range.min.vars() | edge.range.max.vars()
            for v in vars_:
                if not schema[v]:
                    Log.error("{{var}} does not exist in schema", var=v)
        elif edge.domain.dimension:
            vars_ = edge.domain.dimension.fields
            edge.domain.dimension = edge.domain.dimension.copy()
            edge.domain.dimension.fields = [schema[v].es_column for v in vars_]
        elif all(edge.domain.partitions.where):
            vars_ = set()
            for p in edge.domain.partitions:
                vars_ |= p.where.vars()

        try:
            vars_ |= edge.value.vars()
            depths = set(len(c.nested_path) - 1 for v in vars_ for c in schema.leaves(v))
            if -1 in depths:
                Log.error(
                    "Do not know of column {{column}}",
                    column=unwraplist([v for v in vars_ if schema[v] == None])
                )
            if len(depths) > 1:
                Log.error("expression {{expr|quote}} spans tables, can not handle", expr=edge.value)
            max_depth = MAX(depths)
            while len(output) <= max_depth:
                output.append([])
        except Exception as e:
            # USUALLY THE SCHEMA IS EMPTY, SO WE ASSUME THIS IS A SIMPLE QUERY
            max_depth = 0
            output.append([])

        output[max_depth].append(AggsDecoder(edge, query, limit))
    return output


def sort_edges(query, prop):
    ordered_edges = []
    remaining_edges = getattr(query, prop)
    for s in query.sort:
        if not isinstance(s.value, Variable):
            Log.error("can only sort by terms")
        for e in remaining_edges:
            if e.value.var == s.value.var:
                if isinstance(e.domain, SetDomain):
                    pass  # ALREADY SORTED?
                else:
                    e.domain.sort = s.sort
                ordered_edges.append(e)
                remaining_edges.remove(e)
                break
    ordered_edges.extend(remaining_edges)
    return ordered_edges


def es_aggsop(es, frum, query):
    query = query.copy()  # WE WILL MARK UP THIS QUERY
    schema = frum.schema
    select = listwrap(query.select)

    es_query = Data()
    new_select = Data()  # MAP FROM canonical_name (USED FOR NAMES IN QUERY) TO SELECT MAPPING
    formula = []
    for s in select:
        if s.aggregate == "count" and isinstance(s.value, Variable) and s.value.var == ".":
            if schema.query_path == ".":
                s.pull = jx_expression_to_function("doc_count")
            else:
                s.pull = jx_expression_to_function({"coalesce": ["_nested.doc_count", "doc_count", 0]})
        elif isinstance(s.value, Variable):
            if s.aggregate == "count":
                new_select["count_"+literal_field(s.value.var)] += [s]
            else:
                new_select[literal_field(s.value.var)] += [s]
        else:
            formula.append(s)

    for canonical_name, many in new_select.items():
        for s in many:
            es_cols = frum.schema.values(s.value.var)

            if s.aggregate == "count":
                canonical_names = []
                for es_col in es_cols:
                    cn = literal_field(es_col.es_column + "_count")
                    canonical_names.append(cn)
                    es_query.aggs[cn].value_count.field = es_col.es_column
                if len(es_cols) == 1:
                    s.pull = jx_expression_to_function(canonical_names[0] + ".value")
                else:
                    s.pull = jx_expression_to_function({"add": [cn + ".value" for cn in canonical_names]})
            elif s.aggregate == "median":
                if len(es_cols) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")
                # ES USES DIFFERENT METHOD FOR PERCENTILES
                key = literal_field(canonical_name + " percentile")

                es_query.aggs[key].percentiles.field = es_cols[0].es_column
                es_query.aggs[key].percentiles.percents += [50]
                s.pull = jx_expression_to_function(key + ".values.50\.0")
            elif s.aggregate == "percentile":
                if len(es_cols) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")
                # ES USES DIFFERENT METHOD FOR PERCENTILES
                key = literal_field(canonical_name + " percentile")
                if isinstance(s.percentile, text_type) or s.percetile < 0 or 1 < s.percentile:
                    Log.error("Expecting percentile to be a float from 0.0 to 1.0")
                percent = Math.round(s.percentile * 100, decimal=6)

                es_query.aggs[key].percentiles.field = es_cols[0].es_column
                es_query.aggs[key].percentiles.percents += [percent]
                s.pull = jx_expression_to_function(key + ".values." + literal_field(text_type(percent)))
            elif s.aggregate == "cardinality":
                canonical_names = []
                for es_col in es_cols:
                    cn = literal_field(es_col.es_column + "_cardinality")
                    canonical_names.append(cn)
                    es_query.aggs[cn].cardinality.field = es_col.es_column
                if len(es_cols) == 1:
                    s.pull = jx_expression_to_function(canonical_names[0] + ".value")
                else:
                    s.pull = jx_expression_to_function({"add": [cn + ".value" for cn in canonical_names], "default": 0})
            elif s.aggregate == "stats":
                if len(es_cols) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")
                # REGULAR STATS
                stats_name = literal_field(canonical_name)
                es_query.aggs[stats_name].extended_stats.field = es_cols[0].es_column

                # GET MEDIAN TOO!
                median_name = literal_field(canonical_name + "_percentile")
                es_query.aggs[median_name].percentiles.field = es_cols[0].es_column
                es_query.aggs[median_name].percentiles.percents += [50]

                s.pull = get_pull_stats(stats_name, median_name)
            elif s.aggregate == "union":
                pulls = []
                for es_col in es_cols:
                    stats_name = encode_property(es_col.es_column)

                    if es_col.nested_path[0] == ".":
                        es_query.aggs[stats_name] = {"terms": {
                            "field": es_col.es_column,
                            "size": Math.min(s.limit, MAX_LIMIT)
                        }}
                        pulls.append(get_bucket_keys(stats_name))

                    else:
                        es_query.aggs[stats_name] = {
                            "nested": {"path": es_col.nested_path[0]},
                            "aggs": {"_nested": {"terms": {
                                "field": es_col.es_column,
                                "size": Math.min(s.limit, MAX_LIMIT)
                            }}}
                        }
                        pulls.append(get_bucket_keys(stats_name+"._nested"))
                if len(pulls) == 0:
                    s.pull = NULL
                elif len(pulls) == 1:
                    s.pull = pulls[0]
                else:
                    s.pull = lambda row: UNION(
                        p(row)
                        for p in pulls
                    )
            else:
                if len(es_cols) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")

                # PULL VALUE OUT OF THE stats AGGREGATE
                es_query.aggs[literal_field(canonical_name)].extended_stats.field = es_cols[0].es_column
                s.pull = jx_expression_to_function({"coalesce": [literal_field(canonical_name) + "." + aggregates[s.aggregate], s.default]})

    for i, s in enumerate(formula):
        canonical_name = literal_field(s.name)

        if isinstance(s.value, TupleOp):
            if s.aggregate == "count":
                # TUPLES ALWAYS EXIST, SO COUNTING THEM IS EASY
                s.pull = "doc_count"
            else:
                Log.error("{{agg}} is not a supported aggregate over a tuple", agg=s.aggregate)
        elif s.aggregate == "count":
            es_query.aggs[literal_field(canonical_name)].value_count.script = s.value.partial_eval().to_ruby(schema).script(schema)
            s.pull = jx_expression_to_function(literal_field(canonical_name) + ".value")
        elif s.aggregate == "median":
            # ES USES DIFFERENT METHOD FOR PERCENTILES THAN FOR STATS AND COUNT
            key = literal_field(canonical_name + " percentile")

            es_query.aggs[key].percentiles.script = s.value.to_ruby(schema).script(schema)
            es_query.aggs[key].percentiles.percents += [50]
            s.pull = jx_expression_to_function(key + ".values.50\.0")
        elif s.aggregate == "percentile":
            # ES USES DIFFERENT METHOD FOR PERCENTILES THAN FOR STATS AND COUNT
            key = literal_field(canonical_name + " percentile")
            percent = Math.round(s.percentile * 100, decimal=6)

            es_query.aggs[key].percentiles.script = s.value.to_ruby(schema).script(schema)
            es_query.aggs[key].percentiles.percents += [percent]
            s.pull = jx_expression_to_function(key + ".values." + literal_field(text_type(percent)))
        elif s.aggregate == "cardinality":
            # ES USES DIFFERENT METHOD FOR CARDINALITY
            key = canonical_name + " cardinality"

            es_query.aggs[key].cardinality.script = s.value.to_ruby(schema).script(schema)
            s.pull = jx_expression_to_function(key + ".value")
        elif s.aggregate == "stats":
            # REGULAR STATS
            stats_name = literal_field(canonical_name)
            es_query.aggs[stats_name].extended_stats.script = s.value.to_ruby(schema).script(schema)

            # GET MEDIAN TOO!
            median_name = literal_field(canonical_name + " percentile")
            es_query.aggs[median_name].percentiles.script = s.value.to_ruby(schema).script(schema)
            es_query.aggs[median_name].percentiles.percents += [50]

            s.pull = get_pull_stats(stats_name, median_name)
        elif s.aggregate=="union":
            # USE TERMS AGGREGATE TO SIMULATE union
            stats_name = literal_field(canonical_name)
            es_query.aggs[stats_name].terms.script_field = s.value.to_ruby(schema).script(schema)
            s.pull = jx_expression_to_function(stats_name + ".buckets.key")
        else:
            # PULL VALUE OUT OF THE stats AGGREGATE
            s.pull = jx_expression_to_function(canonical_name + "." + aggregates[s.aggregate])
            es_query.aggs[canonical_name].extended_stats.script = s.value.to_ruby(schema).script(schema)

    decoders = get_decoders_by_depth(query)
    start = 0

    #<TERRIBLE SECTION> THIS IS WHERE WE WEAVE THE where CLAUSE WITH nested
    split_where = split_expression_by_depth(query.where, schema=frum.schema)

    if len(split_field(frum.name)) > 1:
        if any(split_where[2::]):
            Log.error("Where clause is too deep")

        for d in decoders[1]:
            es_query = d.append_query(es_query, start)
            start += d.num_columns

        if split_where[1]:
            #TODO: INCLUDE FILTERS ON EDGES
            filter_ = AndOp("and", split_where[1]).to_esfilter(schema)
            es_query = Data(
                aggs={"_filter": set_default({"filter": filter_}, es_query)}
            )

        es_query = wrap({
            "aggs": {"_nested": set_default(
                {
                    "nested": {
                        "path": schema.query_path
                    }
                },
                es_query
            )}
        })
    else:
        if any(split_where[1::]):
            Log.error("Where clause is too deep")

    if decoders:
        for d in jx.reverse(decoders[0]):
            es_query = d.append_query(es_query, start)
            start += d.num_columns

    if split_where[0]:
        #TODO: INCLUDE FILTERS ON EDGES
        filter = AndOp("and", split_where[0]).to_esfilter(schema)
        es_query = Data(
            aggs={"_filter": set_default({"filter": filter}, es_query)}
        )
    # </TERRIBLE SECTION>

    if not es_query:
        es_query = wrap({"query": {"match_all": {}}})

    es_query.size = 0

    with Timer("ES query time") as es_duration:
        result = es_post(es, es_query, query.limit)

    try:
        format_time = Timer("formatting")
        with format_time:
            decoders = [d for ds in decoders for d in ds]
            result.aggregations.doc_count = coalesce(result.aggregations.doc_count, result.hits.total)  # IT APPEARS THE OLD doc_count IS GONE

            formatter, groupby_formatter, aggop_formatter, mime_type = format_dispatch[query.format]
            if query.edges:
                output = formatter(decoders, result.aggregations, start, query, select)
            elif query.groupby:
                output = groupby_formatter(decoders, result.aggregations, start, query, select)
            else:
                output = aggop_formatter(decoders, result.aggregations, start, query, select)

        output.meta.timing.formatting = format_time.duration
        output.meta.timing.es_search = es_duration.duration
        output.meta.content_type = mime_type
        output.meta.es_query = es_query
        return output
    except Exception as e:
        if query.format not in format_dispatch:
            Log.error("Format {{format|quote}} not supported yet", format=query.format, cause=e)
        Log.error("Some problem", cause=e)


EMPTY = {}
EMPTY_LIST = []


def get_bucket_keys(stats_name):
    buckets = jx_expression_to_function(stats_name + ".buckets")
    def output(row):
        return [b['key'] for b in listwrap(buckets(row))]
    return output


def drill(agg):
    deeper = agg.get("_filter") or agg.get("_nested")
    while deeper:
        agg = deeper
        deeper = agg.get("_filter") or agg.get("_nested")
    return agg

def aggs_iterator(aggs, decoders, coord=True):
    """
    DIG INTO ES'S RECURSIVE aggs DATA-STRUCTURE:
    RETURN AN ITERATOR OVER THE EFFECTIVE ROWS OF THE RESULTS

    :param aggs: ES AGGREGATE OBJECT
    :param decoders:
    :param coord: TURN ON LOCAL COORDINATE LOOKUP
    """
    depth = max(d.start + d.num_columns for d in decoders)

    def _aggs_iterator(agg, d):
        agg = drill(agg)

        if d > 0:
            for k, v in agg.items():
                if k == "_match":
                    v = drill(v)
                    for i, b in enumerate(v.get("buckets", EMPTY_LIST)):
                        b["_index"] = i
                        for a, parts in _aggs_iterator(b, d - 1):
                            yield a, parts + (b,)
                elif k == "_other":
                    for b in v.get("buckets", EMPTY_LIST):
                        for a, parts in _aggs_iterator(b, d - 1):
                            yield a, parts + (Null,)
                elif k == "_missing":
                    b = drill(v)
                    for a, parts in _aggs_iterator(b, d - 1):
                        yield a, parts + (b,)
                elif k.startswith("_join_"):
                    v["key"] = int(k[6:])
                    for a, parts in _aggs_iterator(v, d - 1):
                        yield a, parts + (v,)
        else:
            for k, v in agg.items():
                if k == "_match":
                    v = drill(v)
                    for i, b in enumerate(v.get("buckets", EMPTY_LIST)):
                        b["_index"] = i
                        yield b, (b,)
                elif k == "_other":
                    for b in v.get("buckets", EMPTY_LIST):
                        yield b, (Null,)
                elif k == "_missing":
                    b = drill(v,)
                    yield b, (v,)
                elif k.startswith("_join_"):
                    v["_index"] = int(k[6:])
                    yield v, (v,)

    if coord:
        for a, parts in _aggs_iterator(unwrap(aggs), depth - 1):
            coord = tuple(d.get_index(parts) for d in decoders)
            yield parts, coord, a
    else:
        for a, parts in _aggs_iterator(unwrap(aggs), depth - 1):
            yield parts, None, a



def count_dim(aggs, decoders):
    if any(isinstance(d, (DefaultDecoder, DimFieldListDecoder, ObjectDecoder)) for d in decoders):
        # ENUMERATE THE DOMAINS, IF UNKNOWN AT QUERY TIME
        for row, coord, agg in aggs_iterator(aggs, decoders, coord=False):
            for d in decoders:
                d.count(row)
        for d in decoders:
            d.done_count()
    new_edges = wrap([d.edge for d in decoders])
    return new_edges


format_dispatch = {}
from jx_elasticsearch.es14.format import format_cube

_ = format_cube


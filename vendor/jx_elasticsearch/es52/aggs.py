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

from collections import deque

from jx_base.domains import SetDomain
from jx_base.expressions import NULL, TupleOp, Variable as Variable_
from jx_base.query import DEFAULT_LIMIT
from jx_base.language import is_op
from jx_elasticsearch import post as es_post
from jx_elasticsearch.es52.decoders import AggsDecoder
from jx_elasticsearch.es52.es_query import Aggs, ComplexAggs, ExprAggs, FilterAggs, NestedAggs, TermsAggs, simplify
from jx_elasticsearch.es52.expressions import AndOp, ES52, split_expression_by_path
from jx_elasticsearch.es52.painless import Painless
from jx_elasticsearch.es52.setop import get_pull_stats
from jx_elasticsearch.es52.util import aggregates
from jx_python import jx
from jx_python.expressions import jx_expression_to_function
from mo_dots import Data, Null, coalesce, join_field, listwrap, literal_field, unwrap, unwraplist, wrap
from mo_future import first, is_text, text_type
from mo_json import EXISTS, NESTED, OBJECT
from mo_json.typed_encoder import encode_property
from mo_logs import Log
from mo_logs.strings import expand_template, quote
import mo_math
from mo_times.timer import Timer

DEBUG = False

COMPARE_TUPLE = """
(a, b)->{
    int i=0;
    for (dummy in a){  //ONLY THIS FOR LOOP IS ACCEPTED (ALL OTHER FORMS THROW NullPointerException)
        if (a[i]==null){
            if (b[i]==null){
                return 0; 
            }else{
                return -1*({{dir}});
            }//endif
        }else if (b[i]==null) return {{dir}};

        if (a[i]!=b[i]) {
            if (a[i] instanceof Boolean){
                if (b[i] instanceof Boolean){
                    int cmp = Boolean.compare(a[i], b[i]);
                    if (cmp != 0) return cmp;
                } else {
                    return -1;
                }//endif                    
            }else if (a[i] instanceof Number) {
                if (b[i] instanceof Boolean) {
                    return 1                
                } else if (b[i] instanceof Number) {
                    int cmp = Double.compare(a[i], b[i]);
                    if (cmp != 0) return cmp;
                } else {
                    return -1;
                }//endif
            }else {
                if (b[i] instanceof Boolean) {
                    return 1;
                } else if (b[i] instanceof Number) {
                    return 1;
                } else {
                    int cmp = ((String)a[i]).compareTo((String)b[i]);
                    if (cmp != 0) return cmp;
                }//endif
            }//endif
        }//endif
        i=i+1;
    }//for
    return 0;
}
"""


MAX_OF_TUPLE = """
(Object[])Arrays.asList(new Object[]{{{expr1}}, {{expr2}}}).stream().{{op}}("""+COMPARE_TUPLE+""").get()
"""


def is_aggsop(es, query):
    if query.edges or query.groupby or any(a != None and a != "none" for a in listwrap(query.select).aggregate):
        return True
    return False


def get_decoders_by_path(query):
    """
    RETURN MAP FROM QUERY PATH TO LIST OF DECODER ARRAYS

    :param query:
    :return:
    """
    schema = query.frum.schema
    output = Data()

    if query.edges:
        if query.sort and query.format != "cube":
            # REORDER EDGES/GROUPBY TO MATCH THE SORT
            query.edges = sort_edges(query, "edges")
    elif query.groupby:
        if query.sort and query.format != "cube":
            query.groupby = sort_edges(query, "groupby")

    for edge in wrap(coalesce(query.edges, query.groupby, [])):
        limit = coalesce(edge.domain.limit, query.limit, DEFAULT_LIMIT)
        if edge.value != None and not edge.value is NULL:
            edge = edge.copy()
            vars_ = edge.value.vars()
            for v in vars_:
                if not schema.leaves(v.var):
                    Log.error("{{var}} does not exist in schema", var=v)
        elif edge.range:
            vars_ = edge.range.min.vars() | edge.range.max.vars()
            for v in vars_:
                if not schema[v.var]:
                    Log.error("{{var}} does not exist in schema", var=v)
        elif edge.domain.dimension:
            vars_ = edge.domain.dimension.fields
            edge.domain.dimension = edge.domain.dimension.copy()
            edge.domain.dimension.fields = [schema[v].es_column for v in vars_]
        elif all(edge.domain.partitions.where):
            vars_ = set()
            for p in edge.domain.partitions:
                vars_ |= p.where.vars()

        vars_ |= edge.value.vars()
        depths = set(c.nested_path[0] for v in vars_ for c in schema.leaves(v.var))
        if not depths:
            Log.error(
                "Do not know of column {{column}}",
                column=unwraplist([v for v in vars_ if schema[v] == None])
            )
        if len(depths) > 1:
            Log.error("expression {{expr|quote}} spans tables, can not handle", expr=edge.value)

        decoder = AggsDecoder(edge, query, limit)
        output[literal_field(first(depths))] += [decoder]
    return output


def sort_edges(query, prop):
    ordered_edges = []
    remaining_edges = getattr(query, prop)
    for s in jx.reverse(query.sort):
        for e in remaining_edges:
            if e.value == s.value:
                if isinstance(e.domain, SetDomain):
                    pass  # ALREADY SORTED?
                else:
                    e.domain.sort = s.sort
                ordered_edges.append(e)
                remaining_edges.remove(e)
                break
        else:
            Log.error("Can not sort by {{expr}}, can only sort by an existing edge expression", expr=s.value)

    ordered_edges.extend(remaining_edges)
    for i, o in enumerate(ordered_edges):
        o.dim = i  # REORDER THE EDGES
    return ordered_edges


def es_aggsop(es, frum, query):
    query = query.copy()  # WE WILL MARK UP THIS QUERY
    schema = frum.schema
    query_path = schema.query_path[0]
    select = listwrap(query.select)

    new_select = Data()  # MAP FROM canonical_name (USED FOR NAMES IN QUERY) TO SELECT MAPPING
    formula = []
    for s in select:
        if is_op(s.value, Variable_):
            s.query_path = query_path
            if s.aggregate == "count":
                new_select["count_"+literal_field(s.value.var)] += [s]
            else:
                new_select[literal_field(s.value.var)] += [s]
        elif s.aggregate:
            split_select = split_expression_by_path(s.value, schema, lang=Painless)
            for si_key, si_value in split_select.items():
                if si_value:
                    if s.query_path:
                        Log.error("can not handle more than one depth per select")
                    s.query_path = si_key
            formula.append(s)

    acc = Aggs()
    for _, many in new_select.items():
        for s in many:
            canonical_name = s.name
            if s.aggregate in ("value_count", "count"):
                columns = frum.schema.values(s.value.var, exclude_type=(OBJECT, NESTED))
            else:
                columns = frum.schema.values(s.value.var)

            if s.aggregate == "count":
                canonical_names = []
                for column in columns:
                    es_name = column.es_column + "_count"
                    if column.jx_type == EXISTS:
                        if column.nested_path[0] == query_path:
                            canonical_names.append("doc_count")
                            acc.add(NestedAggs(column.nested_path[0]).add(
                                ComplexAggs(s)
                            ))
                    else:
                        canonical_names.append("value")
                        acc.add(NestedAggs(column.nested_path[0]).add(
                            ExprAggs(es_name, {"value_count": {"field": column.es_column}}, s)
                        ))
                if len(canonical_names) == 1:
                    s.pull = jx_expression_to_function(canonical_names[0])
                else:
                    s.pull = jx_expression_to_function({"add": canonical_names})
            elif s.aggregate == "median":
                if len(columns) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")
                # ES USES DIFFERENT METHOD FOR PERCENTILES
                key = canonical_name + " percentile"
                acc.add(ExprAggs(key, {"percentiles": {
                    "field": first(columns).es_column,
                    "percents": [50]
                }}, s))
                s.pull = jx_expression_to_function("values.50\\.0")
            elif s.aggregate == "percentile":
                if len(columns) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")
                # ES USES DIFFERENT METHOD FOR PERCENTILES
                key = canonical_name + " percentile"
                if is_text(s.percentile) or s.percetile < 0 or 1 < s.percentile:
                    Log.error("Expecting percentile to be a float from 0.0 to 1.0")
                percent = mo_math.round(s.percentile * 100, decimal=6)

                acc.add(ExprAggs(key, {"percentiles": {
                    "field": first(columns).es_column,
                    "percents": [percent],
                    "tdigest": {"compression": 2}
                }}, s))
                s.pull = jx_expression_to_function(join_field(["values", text_type(percent)]))
            elif s.aggregate == "cardinality":
                for column in columns:
                    path = column.es_column + "_cardinality"
                    acc.add(ExprAggs(path, {"cardinality": {"field": column.es_column}}, s))
                s.pull = jx_expression_to_function("value")
            elif s.aggregate == "stats":
                if len(columns) > 1:
                    Log.error("Do not know how to count columns with more than one type (script probably)")
                # REGULAR STATS
                stats_name = literal_field(canonical_name)
                complex = ComplexAggs(s).add(ExprAggs(canonical_name, {"extended_stats": {"field": first(columns).es_column}}, None))
                # GET MEDIAN TOO!
                median_name = literal_field(canonical_name + "_percentile")
                complex.add(ExprAggs(canonical_name + "_percentile", {"percentiles": {
                    "field": first(columns).es_column,
                    "percents": [50]
                }}, None))

                acc.add(complex)
                s.pull = get_pull_stats(stats_name, median_name)
            elif s.aggregate == "union":
                for column in columns:
                    script = {"scripted_metric": {
                        'init_script': 'params._agg.terms = new HashSet()',
                        'map_script': 'for (v in doc['+quote(column.es_column)+'].values) params._agg.terms.add(v);',
                        'combine_script': 'return params._agg.terms.toArray()',
                        'reduce_script': 'HashSet output = new HashSet(); for (a in params._aggs) { if (a!=null) for (v in a) {output.add(v)} } return output.toArray()',
                    }}
                    stats_name = column.es_column
                    acc.add(NestedAggs(column.nested_path[0]).add(ExprAggs(stats_name, script, s)))
                s.pull = jx_expression_to_function("value")
            elif s.aggregate == "count_values":
                # RETURN MAP FROM VALUE TO THE NUMBER OF TIMES FOUND IN THE DOCUMENTS
                # NOT A NESTED DOC, RATHER A MULTIVALUE FIELD
                for column in columns:
                    script = {"scripted_metric": {
                        'params': {"_agg": {}},
                        'init_script': 'params._agg.terms = new HashMap()',
                        'map_script': 'for (v in doc['+quote(column.es_column)+'].values) params._agg.terms.put(v, Optional.ofNullable(params._agg.terms.get(v)).orElse(0)+1);',
                        'combine_script': 'return params._agg.terms',
                        'reduce_script': '''
                            HashMap output = new HashMap(); 
                            for (agg in params._aggs) {
                                if (agg!=null){
                                    for (e in agg.entrySet()) {
                                        String key = String.valueOf(e.getKey());
                                        output.put(key, e.getValue() + Optional.ofNullable(output.get(key)).orElse(0));
                                    } 
                                }
                            } 
                            return output;
                        '''
                    }}
                    stats_name = encode_property(column.es_column)
                    acc.add(NestedAggs(column.nested_path[0]).add(ExprAggs(stats_name, script, s)))
                s.pull = jx_expression_to_function("value")
            else:
                if not columns:
                    s.pull = jx_expression_to_function(NULL)
                else:
                    for c in columns:
                        acc.add(NestedAggs(c.nested_path[0]).add(
                            ExprAggs(canonical_name, {"extended_stats": {"field": c.es_column}}, s)
                        ))
                    s.pull = jx_expression_to_function(aggregates[s.aggregate])

    for i, s in enumerate(formula):
        s_path = [k for k, v in split_expression_by_path(s.value, schema=schema, lang=Painless).items() if v]
        if len(s_path) == 0:
            # FOR CONSTANTS
            nest = NestedAggs(query_path)
            acc.add(nest)
        elif len(s_path) == 1:
            nest = NestedAggs(first(s_path))
            acc.add(nest)
        else:
            Log.error("do not know how to handle")

        canonical_name = s.name
        if is_op(s.value, TupleOp):
            if s.aggregate == "count":
                # TUPLES ALWAYS EXIST, SO COUNTING THEM IS EASY
                s.pull = jx_expression_to_function("doc_count")
            elif s.aggregate in ('max', 'maximum', 'min', 'minimum'):
                if s.aggregate in ('max', 'maximum'):
                    dir = 1
                    op = "max"
                else:
                    dir = -1
                    op = 'min'

                nully = Painless[TupleOp([NULL]*len(s.value.terms))].partial_eval().to_es_script(schema)
                selfy = text_type(Painless[s.value].partial_eval().to_es_script(schema))

                script = {"scripted_metric": {
                    'init_script': 'params._agg.best = ' + nully + ';',
                    'map_script': 'params._agg.best = ' + expand_template(MAX_OF_TUPLE, {"expr1": "params._agg.best", "expr2": selfy, "dir": dir, "op": op}) + ";",
                    'combine_script': 'return params._agg.best',
                    'reduce_script': 'return params._aggs.stream().'+op+'(' + expand_template(COMPARE_TUPLE, {"dir": dir, "op": op}) + ').get()',
                }}
                nest.add(NestedAggs(query_path).add(
                    ExprAggs(canonical_name, script, s)
                ))
                s.pull = jx_expression_to_function("value")
            else:
                Log.error("{{agg}} is not a supported aggregate over a tuple", agg=s.aggregate)
        elif s.aggregate == "count":
            nest.add(ExprAggs(canonical_name, {"value_count": {"script": text_type(Painless[s.value].partial_eval().to_es_script(schema))}}, s))
            s.pull = jx_expression_to_function("value")
        elif s.aggregate == "median":
            # ES USES DIFFERENT METHOD FOR PERCENTILES THAN FOR STATS AND COUNT
            key = literal_field(canonical_name + " percentile")
            nest.add(ExprAggs(key, {"percentiles": {
                "script": text_type(Painless[s.value].to_es_script(schema)),
                "percents": [50]
            }}, s))
            s.pull = jx_expression_to_function(join_field(["50.0"]))
        elif s.aggregate == "percentile":
            # ES USES DIFFERENT METHOD FOR PERCENTILES THAN FOR STATS AND COUNT
            key = literal_field(canonical_name + " percentile")
            percent = mo_math.round(s.percentile * 100, decimal=6)
            nest.add(ExprAggs(key, {"percentiles": {
                "script": text_type(Painless[s.value].to_es_script(schema)),
                "percents": [percent]
            }}, s))
            s.pull = jx_expression_to_function(join_field(["values", text_type(percent)]))
        elif s.aggregate == "cardinality":
            # ES USES DIFFERENT METHOD FOR CARDINALITY
            key = canonical_name + " cardinality"
            nest.add(ExprAggs(key, {"cardinality": {"script": text_type(Painless[s.value].to_es_script(schema))}}, s))
            s.pull = jx_expression_to_function("value")
        elif s.aggregate == "stats":
            # REGULAR STATS
            stats_name = canonical_name
            nest.add(ComplexAggs(s).add(ExprAggs(stats_name, {"extended_stats": {"script": text_type(Painless[s.value].to_es_script(schema))}}, None)))

            # GET MEDIAN TOO!
            median_name = canonical_name + " percentile"
            nest.add(ExprAggs(median_name, {"percentiles": {
                "script": text_type(Painless[s.value].to_es_script(schema)),
                "percents": [50]
            }}, s))
            s.pull = get_pull_stats(None, stats_name, median_name)
        elif s.aggregate == "union":
            # USE TERMS AGGREGATE TO SIMULATE union
            nest.add(TermsAggs(canonical_name, {"script_field": text_type(Painless[s.value].to_es_script(schema))}, s))
            s.pull = jx_expression_to_function("key")
        else:
            # PULL VALUE OUT OF THE stats AGGREGATE
            s.pull = jx_expression_to_function(aggregates[s.aggregate])
            nest.add(ExprAggs(canonical_name, {"extended_stats": {"script": text_type(Painless[s.value].to_es_script(schema))}}, s))

    acc = NestedAggs(query_path).add(acc)
    split_decoders = get_decoders_by_path(query)
    split_wheres = split_expression_by_path(query.where, schema=frum.schema, lang=ES52)

    start = 0
    decoders = [None] * (len(query.edges) + len(query.groupby))
    paths = list(reversed(sorted(split_wheres.keys() | split_decoders.keys())))
    for path in paths:
        literal_path = literal_field(path)
        decoder = split_decoders[literal_path]
        where = split_wheres[literal_path]

        for d in decoder:
            decoders[d.edge.dim] = d
            acc = d.append_query(path, acc)
            start += d.num_columns

        if where:
            acc = FilterAggs("_filter", AndOp(where), None).add(acc)
        acc = NestedAggs(path).add(acc)

    acc = NestedAggs('.').add(acc)
    acc = simplify(acc)
    es_query = wrap(acc.to_es(schema))

    es_query.size = 0

    with Timer("ES query time", silent=not DEBUG) as es_duration:
        result = es_post(es, es_query, query.limit)

    try:
        format_time = Timer("formatting", silent=not DEBUG)
        with format_time:
            # result.aggregations.doc_count = coalesce(result.aggregations.doc_count, result.hits.total)  # IT APPEARS THE OLD doc_count IS GONE
            aggs = unwrap(result.aggregations)

            formatter, groupby_formatter, aggop_formatter, mime_type = format_dispatch[query.format]
            if query.edges:
                output = formatter(aggs, acc, query, decoders, select)
            elif query.groupby:
                output = groupby_formatter(aggs, acc, query, decoders, select)
            else:
                output = aggop_formatter(aggs, acc, query, decoders, select)

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


def drill(agg):
    while True:
        deeper = agg.get("_filter")
        if deeper:
            agg = deeper
            continue
        return agg


def _children(agg, children):
    for child in children:
        name = child.name
        v = agg[name]
        if name == "_match":
            for i, b in enumerate(v.get("buckets", EMPTY_LIST)):
                yield i, b, child, b
        elif name.startswith("_match"):
            i = int(name[6:])
            yield i, v, child, v
        elif name.startswith("_missing"):
            if len(name) == 8:
                i = None
            else:
                i = int(name[8:])
            yield None, v, child, v
        else:
            yield None, v, child, None


def aggs_iterator(aggs, es_query, decoders, give_me_zeros=False):
    """
    DIG INTO ES'S RECURSIVE aggs DATA-STRUCTURE:
    RETURN AN ITERATOR OVER THE EFFECTIVE ROWS OF THE RESULTS

    :param aggs: ES AGGREGATE OBJECT
    :param es_query: THE ABSTRACT ES QUERY WE WILL TRACK ALONGSIDE aggs
    :param decoders: TO CONVERT PARTS INTO COORDINATES
    """
    coord = [0] * len(decoders)
    parts = deque()
    stack = []

    gen = _children(aggs, es_query.children)
    while True:
        try:
            index, c_agg, c_query, part = gen.next()
        except StopIteration:
            try:
                gen = stack.pop()
            except IndexError:
                return
            parts.popleft()
            continue

        if c_agg.get('doc_count') == 0 and not give_me_zeros:
            continue
        parts.appendleft(part)
        for d in c_query.decoders:
            coord[d.edge.dim] = d.get_index(tuple(p for p in parts if p is not None), c_query, index)

        children = c_query.children
        selects = c_query.selects
        if selects or not children:
            parts.popleft()  # c_agg WAS ON TOP
            yield (
                tuple(p for p in parts if p is not None),
                tuple(coord),
                c_agg,
                selects
            )
            continue

        stack.append(gen)
        gen = _children(c_agg, children)


def count_dim(aggs, es_query, decoders):
    if not any(hasattr(d, "done_count") for d in decoders):
        return [d.edge for d in decoders]

    def _count_dim(parts, aggs, es_query):
        children = es_query.children
        if not children:
            return

        for child in children:
            name = child.name
            agg = aggs[name]
            if agg.get('doc_count') == 0:
                continue
            elif name == "_match":
                for i, b in enumerate(agg.get("buckets", EMPTY_LIST)):
                    if not b.get('doc_count'):
                        continue
                    b["_index"] = i
                    new_parts = (b,) + parts
                    for d in child.decoders:
                        d.count(new_parts)
                    _count_dim(new_parts, b, child)
            elif name.startswith("_missing"):
                new_parts = (agg,) + parts
                for d in child.decoders:
                    d.count(new_parts)
                _count_dim(new_parts, agg, child)
            else:
                _count_dim(parts, agg, child)

    _count_dim(tuple(), aggs, es_query)
    for d in decoders:
        done_count = getattr(d, "done_count", Null)
        done_count()
    return [d.edge for d in decoders]


format_dispatch = {}

from jx_elasticsearch.es52.format import format_cube
_ = format_cube


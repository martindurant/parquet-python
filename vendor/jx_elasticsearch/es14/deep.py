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

from jx_base import STRUCT, NESTED, EXISTS
from jx_base.expressions import NULL
from jx_base.query import DEFAULT_LIMIT
from jx_elasticsearch.es09.util import post as es_post
from jx_elasticsearch.es14.expressions import split_expression_by_depth, AndOp, Variable, LeavesOp
from jx_elasticsearch.es14.setop import format_dispatch, get_pull_function, get_pull
from jx_elasticsearch.es14.util import jx_sort_to_es_sort, es_query_template
from jx_python.expressions import compile_expression, jx_expression_to_function
from mo_dots import split_field, FlatList, listwrap, literal_field, coalesce, Data, concat_field, set_default, relative_field, startswith_field
from mo_json.typed_encoder import untype_path, EXISTS_TYPE
from mo_logs import Log
from mo_threads import Thread
from mo_times.timer import Timer
from pyLibrary import convert

EXPRESSION_PREFIX = "_expr."

_ = convert


def is_deepop(es, query):
    if query.edges or query.groupby:
        return False
    if all(s.aggregate not in (None, "none") for s in listwrap(query.select)):
        return False
    if len(split_field(query.frum.name)) > 1:
        return True

    # ASSUME IT IS NESTED IF WE ARE ASKING FOR NESTED COLUMNS
    # vars_ = query_get_all_vars(query)
    # columns = query.frum.get_columns()
    # if any(c for c in columns if len(c.nested_path) != 1 and c.name in vars_):
    #    return True
    return False


def es_deepop(es, query):
    schema = query.frum.schema
    columns = schema.columns
    query_path = schema.query_path

    # TODO: FIX THE GREAT SADNESS CAUSED BY EXECUTING post_expressions
    # THE EXPRESSIONS SHOULD BE PUSHED TO THE CONTAINER:  ES ALLOWS
    # {"inner_hit":{"script_fields":[{"script":""}...]}}, BUT THEN YOU
    # LOOSE "_source" BUT GAIN "fields", FORCING ALL FIELDS TO BE EXPLICIT
    post_expressions = {}
    es_query, es_filters = es_query_template(query_path)

    # SPLIT WHERE CLAUSE BY DEPTH
    wheres = split_expression_by_depth(query.where, schema)
    for i, f in enumerate(es_filters):
        script = AndOp("and", wheres[i]).partial_eval().to_esfilter(schema)
        set_default(f, script)

    if not wheres[1]:
        # WITHOUT NESTED CONDITIONS, WE MUST ALSO RETURN DOCS WITH NO NESTED RECORDS
        more_filter = {
            "and": [
                es_filters[0],
                {"missing": {"field": untype_path(query_path) + "." + EXISTS_TYPE}}
            ]
        }
    else:
        more_filter = None

    es_query.size = coalesce(query.limit, DEFAULT_LIMIT)

    # es_query.sort = jx_sort_to_es_sort(query.sort)
    map_to_es_columns = schema.map_to_es()
    # {c.names["."]: c.es_column for c in schema.leaves(".")}
    query_for_es = query.map(map_to_es_columns)
    es_query.sort = jx_sort_to_es_sort(query_for_es.sort, schema)

    es_query.fields = []

    is_list = isinstance(query.select, list)
    new_select = FlatList()

    i = 0
    for s in listwrap(query.select):
        if isinstance(s.value, LeavesOp) and isinstance(s.value.term, Variable):
            # IF THERE IS A *, THEN INSERT THE EXTRA COLUMNS
            leaves = schema.leaves(s.value.term.var)
            col_names = set()
            for c in leaves:
                if c.nested_path[0] == ".":
                    if c.type == NESTED:
                        continue
                    es_query.fields += [c.es_column]
                c_name = untype_path(c.names[query_path])
                col_names.add(c_name)
                new_select.append({
                    "name": concat_field(s.name, c_name),
                    "nested_path": c.nested_path[0],
                    "put": {"name": concat_field(s.name, literal_field(c_name)), "index": i, "child": "."},
                    "pull": get_pull_function(c)
                })
                i += 1

            # REMOVE DOTS IN PREFIX IF NAME NOT AMBIGUOUS
            for n in new_select:
                if n.name.startswith("..") and n.name.lstrip(".") not in col_names:
                    n.put.name = n.name = n.name.lstrip(".")
                    col_names.add(n.name)
        elif isinstance(s.value, Variable):
            net_columns = schema.leaves(s.value.var)
            if not net_columns:
                new_select.append({
                    "name": s.name,
                    "nested_path": ".",
                    "put": {"name": s.name, "index": i, "child": "."},
                    "pull": NULL
                })
            else:
                for n in net_columns:
                    pull = get_pull_function(n)
                    if n.nested_path[0] == ".":
                        if n.type == NESTED:
                            continue
                        es_query.fields += [n.es_column]

                    # WE MUST FIGURE OUT WHICH NAMESSPACE s.value.var IS USING SO WE CAN EXTRACT THE child
                    for np in n.nested_path:
                        c_name = untype_path(n.names[np])
                        if startswith_field(c_name, s.value.var):
                            child = relative_field(c_name, s.value.var)
                            break
                    else:
                        child = relative_field(untype_path(n.names[n.nested_path[0]]), s.value.var)

                    new_select.append({
                        "name": s.name,
                        "pull": pull,
                        "nested_path": n.nested_path[0],
                        "put": {
                            "name": s.name,
                            "index": i,
                            "child": child
                        }
                    })
            i += 1
        else:
            expr = s.value
            for v in expr.vars():
                for c in schema[v]:
                    if c.nested_path[0] == ".":
                        es_query.fields += [c.es_column]
                    # else:
                    #     Log.error("deep field not expected")

            pull_name = EXPRESSION_PREFIX + s.name
            map_to_local = {untype_path(k): get_pull(cc) for k, c in schema.lookup.items() for cc in c if cc.type not in STRUCT}
            pull = jx_expression_to_function(pull_name)
            post_expressions[pull_name] = compile_expression(expr.map(map_to_local).to_python())

            new_select.append({
                "name": s.name if is_list else ".",
                "pull": pull,
                "value": expr.__data__(),
                "put": {"name": s.name, "index": i, "child": "."}
            })
            i += 1

    # <COMPLICATED> ES needs two calls to get all documents
    more = []
    def get_more(please_stop):
        more.append(es_post(
            es,
            Data(
                query={"filtered": {"filter": more_filter}},
                fields=es_query.fields
            ),
            query.limit
        ))
    if more_filter:
        need_more = Thread.run("get more", target=get_more)

    with Timer("call to ES") as call_timer:
        data = es_post(es, es_query, query.limit)

    # EACH A HIT IS RETURNED MULTIPLE TIMES FOR EACH INNER HIT, WITH INNER HIT INCLUDED
    def inners():
        for t in data.hits.hits:
            for i in t.inner_hits[literal_field(query_path)].hits.hits:
                t._inner = i._source
                for k, e in post_expressions.items():
                    t[k] = e(t)
                yield t
        if more_filter:
            Thread.join(need_more)
            for t in more[0].hits.hits:
                yield t
    #</COMPLICATED>

    try:
        formatter, groupby_formatter, mime_type = format_dispatch[query.format]

        output = formatter(inners(), new_select, query)
        output.meta.timing.es = call_timer.duration
        output.meta.content_type = mime_type
        output.meta.es_query = es_query
        return output
    except Exception as e:
        Log.error("problem formatting", e)



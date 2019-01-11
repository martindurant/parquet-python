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

from mo_future import is_text, is_binary
from jx_base.expressions import LeavesOp, NULL, Variable
from jx_base.query import DEFAULT_LIMIT
from jx_base.language import is_op
from jx_elasticsearch import post as es_post
from jx_elasticsearch.es52.expressions import AndOp, ES52, split_expression_by_depth
from jx_elasticsearch.es52.setop import format_dispatch, get_pull, get_pull_function
from jx_elasticsearch.es52.util import es_query_template, jx_sort_to_es_sort
from jx_python.expressions import jx_expression_to_function
from mo_dots import Data, FlatList, coalesce, concat_field, is_list as is_list_, listwrap, literal_field, relative_field, set_default, split_field, startswith_field, unwrap, wrap
from mo_future import zip_longest
from mo_json import NESTED
from mo_json.typed_encoder import untype_path
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
    query_path = schema.query_path[0]

    # TODO: FIX THE GREAT SADNESS CAUSED BY EXECUTING post_expressions
    # THE EXPRESSIONS SHOULD BE PUSHED TO THE CONTAINER:  ES ALLOWS
    # {"inner_hit":{"script_fields":[{"script":""}...]}}, BUT THEN YOU
    # LOOSE "_source" BUT GAIN "fields", FORCING ALL FIELDS TO BE EXPLICIT
    post_expressions = {}
    es_query, es_filters = es_query_template(query_path)

    # SPLIT WHERE CLAUSE BY DEPTH
    wheres = split_expression_by_depth(query.where, schema)
    for f, w in zip_longest(es_filters, wheres):
        script = ES52[AndOp(w)].partial_eval().to_esfilter(schema)
        set_default(f, script)

    if not wheres[1]:
        # INCLUDE DOCS WITH NO NESTED DOCS
        more_filter = {
            "bool": {
                "filter": [AndOp(wheres[0]).partial_eval().to_esfilter(schema)],
                "must_not": {
                    "nested": {
                        "path": query_path,
                        "query": {
                            "match_all": {}
                        }
                    }
                }
            }
        }
    else:
        more_filter = None

    es_query.size = coalesce(query.limit, DEFAULT_LIMIT)

    # es_query.sort = jx_sort_to_es_sort(query.sort)
    map_to_es_columns = schema.map_to_es()
    # {c.name: c.es_column for c in schema.leaves(".")}
    query_for_es = query.map(map_to_es_columns)
    es_query.sort = jx_sort_to_es_sort(query_for_es.sort, schema)

    es_query.stored_fields = []

    is_list = is_list_(query.select)
    selects = wrap([unwrap(s.copy()) for s in listwrap(query.select)])
    new_select = FlatList()

    put_index = 0
    for select in selects:
        if is_op(select.value, LeavesOp) and is_op(select.value.term, Variable):
            # IF THERE IS A *, THEN INSERT THE EXTRA COLUMNS
            leaves = schema.leaves(select.value.term.var)
            col_names = set()
            for c in leaves:
                if c.nested_path[0] == ".":
                    if c.jx_type == NESTED:
                        continue
                    es_query.stored_fields += [c.es_column]
                c_name = untype_path(relative_field(c.name, query_path))
                col_names.add(c_name)
                new_select.append({
                    "name": concat_field(select.name, c_name),
                    "nested_path": c.nested_path[0],
                    "put": {"name": concat_field(select.name, literal_field(c_name)), "index": put_index, "child": "."},
                    "pull": get_pull_function(c)
                })
                put_index += 1

            # REMOVE DOTS IN PREFIX IF NAME NOT AMBIGUOUS
            for n in new_select:
                if n.name.startswith("..") and n.name.lstrip(".") not in col_names:
                    n.put.name = n.name = n.name.lstrip(".")
                    col_names.add(n.name)
        elif is_op(select.value, Variable):
            net_columns = schema.leaves(select.value.var)
            if not net_columns:
                new_select.append({
                    "name": select.name,
                    "nested_path": ".",
                    "put": {"name": select.name, "index": put_index, "child": "."},
                    "pull": NULL
                })
            else:
                for n in net_columns:
                    pull = get_pull_function(n)
                    if n.nested_path[0] == ".":
                        if n.jx_type == NESTED:
                            continue
                        es_query.stored_fields += [n.es_column]

                    # WE MUST FIGURE OUT WHICH NAMESSPACE s.value.var IS USING SO WE CAN EXTRACT THE child
                    for np in n.nested_path:
                        c_name = untype_path(relative_field(n.name, np))
                        if startswith_field(c_name, select.value.var):
                            child = relative_field(c_name, select.value.var)
                            break
                    else:
                        continue
                        # REMOVED BECAUSE SELECTING INNER PROPERTIES IS NOT ALLOWED
                        # child = relative_field(untype_path(relative_field(n.name, n.nested_path[0])), s.value.var)

                    new_select.append({
                        "name": select.name,
                        "pull": pull,
                        "nested_path": n.nested_path[0],
                        "put": {
                            "name": select.name,
                            "index": put_index,
                            "child": child
                        }
                    })
            put_index += 1
        else:
            expr = select.value
            for v in expr.vars():
                for c in schema[v.var]:
                    if c.nested_path[0] == ".":
                        es_query.stored_fields += [c.es_column]
                    # else:
                    #     Log.error("deep field not expected")

            pull_name = EXPRESSION_PREFIX + select.name
            map_to_local = MapToLocal(schema)
            pull = jx_expression_to_function(pull_name)
            post_expressions[pull_name] = jx_expression_to_function(expr.map(map_to_local))

            new_select.append({
                "name": select.name if is_list else ".",
                "pull": pull,
                "value": expr.__data__(),
                "put": {"name": select.name, "index": put_index, "child": "."}
            })
            put_index += 1

    # <COMPLICATED> ES needs two calls to get all documents
    more = []
    def get_more(please_stop):
        more.append(es_post(
            es,
            Data(
                query=more_filter,
                stored_fields=es_query.stored_fields
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
    # </COMPLICATED>

    try:
        formatter, groupby_formatter, mime_type = format_dispatch[query.format]

        output = formatter(inners(), new_select, query)
        output.meta.timing.es = call_timer.duration
        output.meta.content_type = mime_type
        output.meta.es_query = es_query
        return output
    except Exception as e:
        Log.error("problem formatting", e)


class MapToLocal(object):
    """
    MAP FROM RELATIVE/ABSOLUTE NAMESPACE TO PYTHON THAT WILL EXTRACT RESULT
    """
    def __init__(self, map_to_columns):
        self.map_to_columns = map_to_columns

    def __getitem__(self, item):
        return self.get(item)

    def get(self, item):
        cs = self.map_to_columns[item]
        if len(cs) == 0:
            return "Null"
        elif len(cs) == 1:
            return get_pull(cs[0])
        else:
            return "coalesce(" + (",".join(get_pull(c) for c in cs)) + ")"



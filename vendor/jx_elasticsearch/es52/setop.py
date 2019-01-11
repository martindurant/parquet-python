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

from jx_base.domains import ALGEBRAIC
from jx_base.expressions import IDENTITY, LeavesOp, Variable
from jx_base.query import DEFAULT_LIMIT
from jx_base.language import is_op
from jx_elasticsearch import post as es_post
from jx_elasticsearch.es52.expressions import AndOp, ES52, split_expression_by_path
from jx_elasticsearch.es52.painless import Painless
from jx_elasticsearch.es52.util import MATCH_ALL, es_and, es_or, jx_sort_to_es_sort
from jx_python.containers.cube import Cube
from jx_python.expressions import jx_expression_to_function
from mo_collections.matrix import Matrix
from mo_dots import Data, FlatList, coalesce, concat_field, is_data, is_list, join_field, listwrap, literal_field, relative_field, set_default, split_field, unwrap, unwraplist, wrap
from mo_future import first, text_type, transpose
from mo_json import NESTED
from mo_json.typed_encoder import decode_property, unnest_path, untype_path, untyped
from mo_logs import Log
from mo_math import AND, MAX
from mo_times.timer import Timer

format_dispatch = {}


def is_setop(es, query):
    select = listwrap(query.select)

    if not query.edges:
        isDeep = len(split_field(query.frum.name)) > 1  # LOOKING INTO NESTED WILL REQUIRE A SCRIPT
        simpleAgg = AND([s.aggregate in ("count", "none") for s in select])  # CONVERTING esfilter DEFINED PARTS WILL REQUIRE SCRIPT

        # NO EDGES IMPLIES SIMPLER QUERIES: EITHER A SET OPERATION, OR RETURN SINGLE AGGREGATE
        if simpleAgg or isDeep:
            return True
    else:
        isSmooth = AND((e.domain.type in ALGEBRAIC and e.domain.interval == "none") for e in query.edges)
        if isSmooth:
            return True

    return False


def es_setop(es, query):
    schema = query.frum.schema
    query_path = schema.query_path[0]

    split_select = {".": ESSelect('.')}

    def get_select(path):
        es_select = split_select.get(path)
        if not es_select:
            es_select = split_select[path] = ESSelect(path)
        return es_select


    selects = wrap([unwrap(s.copy()) for s in listwrap(query.select)])
    new_select = FlatList()

    put_index = 0
    for select in selects:
        # IF THERE IS A *, THEN INSERT THE EXTRA COLUMNS
        if is_op(select.value, LeavesOp) and is_op(select.value.term, Variable):
            term = select.value.term
            leaves = schema.leaves(term.var)
            for c in leaves:
                full_name = concat_field(select.name, relative_field(untype_path(c.name), term.var))
                if c.jx_type == NESTED:
                    get_select('.').use_source = True
                    new_select.append({
                        "name": full_name,
                        "value": Variable(c.es_column),
                        "put": {"name": literal_field(full_name), "index": put_index, "child": "."},
                        "pull": get_pull_source(c.es_column)
                    })
                    put_index += 1
                else:
                    get_select(c.nested_path[0]).fields.append(c.es_column)
                    new_select.append({
                        "name": full_name,
                        "value": Variable(c.es_column),
                        "put": {"name": literal_field(full_name), "index": put_index, "child": "."}
                    })
                    put_index += 1
        elif is_op(select.value, Variable):
            s_column = select.value.var

            if s_column == ".":
                # PULL ALL SOURCE
                get_select('.').use_source = True
                new_select.append({
                    "name": select.name,
                    "value": select.value,
                    "put": {"name": select.name, "index": put_index, "child": "."},
                    "pull": get_pull_source(".")
                })
                continue

            leaves = schema.leaves(s_column)  # LEAVES OF OBJECT
            # nested_selects = {}
            if leaves:
                if any(c.jx_type == NESTED for c in leaves):
                    # PULL WHOLE NESTED ARRAYS
                    get_select('.').use_source = True
                    for c in leaves:
                        if len(c.nested_path) == 1:  # NESTED PROPERTIES ARE IGNORED, CAPTURED BY THESE FIRST LEVEL PROPERTIES
                            pre_child = join_field(decode_property(n) for n in split_field(c.name))
                            new_select.append({
                                "name": select.name,
                                "value": Variable(c.es_column),
                                "put": {"name": select.name, "index": put_index, "child": untype_path(relative_field(pre_child, s_column))},
                                "pull": get_pull_source(c.es_column)
                            })
                else:
                    # PULL ONLY WHAT'S NEEDED
                    for c in leaves:
                        c_nested_path = c.nested_path[0]
                        if c_nested_path == ".":
                            if c.es_column == "_id":
                                new_select.append({
                                    "name": select.name,
                                    "value": Variable(c.es_column),
                                    "put": {"name": select.name, "index": put_index, "child": "."},
                                    "pull": lambda row: row._id
                                })
                            elif c.jx_type == NESTED:
                                get_select('.').use_source = True
                                pre_child = join_field(decode_property(n) for n in split_field(c.name))
                                new_select.append({
                                    "name": select.name,
                                    "value": Variable(c.es_column),
                                    "put": {"name": select.name, "index": put_index, "child": untype_path(relative_field(pre_child, s_column))},
                                    "pull": get_pull_source(c.es_column)
                                })
                            else:
                                get_select(c_nested_path).fields.append(c.es_column)
                                pre_child = join_field(decode_property(n) for n in split_field(c.name))
                                new_select.append({
                                    "name": select.name,
                                    "value": Variable(c.es_column),
                                    "put": {"name": select.name, "index": put_index, "child": untype_path(relative_field(pre_child, s_column))}
                                })
                        else:
                            es_select = get_select(c_nested_path)
                            es_select.fields.append(c.es_column)

                            child = relative_field(untype_path(relative_field(c.name, schema.query_path[0])), s_column)
                            pull = accumulate_nested_doc(c_nested_path, Variable(relative_field(s_column, unnest_path(c_nested_path))))
                            new_select.append({
                                "name": select.name,
                                "value": select.value,
                                "put": {
                                    "name": select.name,
                                    "index": put_index,
                                    "child": child
                                },
                                "pull": pull
                            })
            else:
                new_select.append({
                    "name": select.name,
                    "value": Variable("$dummy"),
                    "put": {"name": select.name, "index": put_index, "child": "."}
                })
            put_index += 1
        else:
            split_scripts = split_expression_by_path(select.value, schema, lang=Painless)
            for p, script in split_scripts.items():
                es_select = get_select(p)
                es_select.scripts[select.name] = {"script": text_type(Painless[first(script)].partial_eval().to_es_script(schema))}
                new_select.append({
                    "name": select.name,
                    "pull": jx_expression_to_function("fields." + literal_field(select.name)),
                    "put": {"name": select.name, "index": put_index, "child": "."}
                })
                put_index += 1

    for n in new_select:
        if n.pull:
            continue
        elif is_op(n.value, Variable):
            if get_select('.').use_source:
                n.pull = get_pull_source(n.value.var)
            elif n.value == "_id":
                n.pull = jx_expression_to_function("_id")
            else:
                n.pull = jx_expression_to_function(concat_field("fields", literal_field(n.value.var)))
        else:
            Log.error("Do not know what to do")

    split_wheres = split_expression_by_path(query.where, schema, lang=ES52)
    es_query = es_query_proto(query_path, split_select, split_wheres, schema)
    es_query.size = coalesce(query.limit, DEFAULT_LIMIT)
    es_query.sort = jx_sort_to_es_sort(query.sort, schema)

    with Timer("call to ES", silent=True) as call_timer:
        data = es_post(es, es_query, query.limit)

    T = data.hits.hits

    # Log.note("{{output}}", output=T)

    try:
        formatter, groupby_formatter, mime_type = format_dispatch[query.format]

        with Timer("formatter", silent=True):
            output = formatter(T, new_select, query)
        output.meta.timing.es = call_timer.duration
        output.meta.content_type = mime_type
        output.meta.es_query = es_query
        return output
    except Exception as e:
        Log.error("problem formatting", e)


def accumulate_nested_doc(nested_path, expr=IDENTITY):
    """
    :param nested_path: THE PATH USED TO EXTRACT THE NESTED RECORDS
    :param expr: FUNCTION USED ON THE NESTED OBJECT TO GET SPECIFIC VALUE
    :return: THE DE_TYPED NESTED OBJECT ARRAY
    """
    name = literal_field(nested_path)
    def output(doc):
        acc = []
        for h in doc.inner_hits[name].hits.hits:
            i = h._nested.offset
            obj = Data()
            for f, v in h.fields.items():
                local_path = untype_path(relative_field(f, nested_path))
                obj[local_path] = unwraplist(v)
            # EXTEND THE LIST TO THE LENGTH WE REQUIRE
            for _ in range(len(acc), i+1):
                acc.append(None)
            acc[i] = expr(obj)
        return acc
    return output


def format_list(T, select, query=None):
    data = []
    if is_list(query.select):
        for row in T:
            r = Data()
            for s in select:
                v = unwraplist(s.pull(row))
                if v is not None:
                    try:
                        r[s.put.name][s.put.child] = v
                    except Exception as e:
                        Log.error("what's happening here?")
            data.append(r if r else None)
    elif is_op(query.select.value, LeavesOp):
        for row in T:
            r = Data()
            for s in select:
                r[s.put.name][s.put.child] = unwraplist(s.pull(row))
            data.append(r if r else None)
    else:
        for row in T:
            r = None
            for s in select:
                v = unwraplist(s.pull(row))
                if v is None:
                    continue
                if s.put.child == ".":
                    r = v
                else:
                    if r is None:
                        r = Data()
                    r[s.put.child] = v

            data.append(r)

    return Data(
        meta={"format": "list"},
        data=data
    )


def format_table(T, select, query=None):
    data = []
    num_columns = (MAX(select.put.index) + 1)
    for row in T:
        r = [None] * num_columns
        for s in select:
            value = unwraplist(s.pull(row))

            if value == None:
                continue

            index, child = s.put.index, s.put.child
            if child == ".":
                r[index] = value
            else:
                if r[index] is None:
                    r[index] = Data()
                r[index][child] = value

        data.append(r)

    header = [None] * num_columns

    if is_data(query.select) and not is_op(query.select.value, LeavesOp):
        for s in select:
            header[s.put.index] = s.name
    else:
        for s in select:
            if header[s.put.index]:
                continue
            if s.name == ".":
                header[s.put.index] = "."
            else:
                header[s.put.index] = s.name

    return Data(
        meta={"format": "table"},
        header=header,
        data=data
    )


def format_cube(T, select, query=None):
    with Timer("format table"):
        table = format_table(T, select, query)

    if len(table.data) == 0:
        return Cube(
            select,
            edges=[{"name": "rownum", "domain": {"type": "rownum", "min": 0, "max": 0, "interval": 1}}],
            data={h: Matrix(list=[]) for i, h in enumerate(table.header)}
        )

    cols = transpose(*unwrap(table.data))
    return Cube(
        select,
        edges=[{"name": "rownum", "domain": {"type": "rownum", "min": 0, "max": len(table.data), "interval": 1}}],
        data={h: Matrix(list=cols[i]) for i, h in enumerate(table.header)}
    )


set_default(format_dispatch, {
    None: (format_cube, None, "application/json"),
    "cube": (format_cube, None, "application/json"),
    "table": (format_table, None, "application/json"),
    "list": (format_list, None, "application/json")
})


def get_pull(column):
    if column.nested_path[0] == ".":
        return concat_field("fields", literal_field(column.es_column))
    else:
        rel_name = relative_field(column.es_column, column.nested_path[0])
        return concat_field("_inner", rel_name)


def get_pull_function(column):
    return jx_expression_to_function(get_pull(column))


def get_pull_source(es_column):
    def output(row):
        return untyped(row._source[es_column])
    return output


def get_pull_stats(stats_name, median_name):
    return jx_expression_to_function({"select": [
        {"name": "count", "value": join_field([stats_name, "count"])},
        {"name": "sum", "value": join_field([stats_name, "sum"])},
        {"name": "min", "value": join_field([stats_name, "min"])},
        {"name": "max", "value": join_field([stats_name, "max"])},
        {"name": "avg", "value": join_field([stats_name, "avg"])},
        {"name": "sos", "value": join_field([stats_name, "sum_of_squares"])},
        {"name": "std", "value": join_field([stats_name, "std_deviation"])},
        {"name": "var", "value": join_field([stats_name, "variance"])},
        {"name": "median", "value": join_field([median_name, "values", "50.0"])}
    ]})


class ESSelect(object):
    """
    ACCUMULATE THE FIELDS WE ARE INTERESTED IN
    """


    def __init__(self, path):
        self.path = path
        self.use_source = False
        self.fields = []
        self.scripts = {}

    def to_es(self):
        return {
            "_source": self.use_source,
            "stored_fields": self.fields if not self.use_source else None,
            "script_fields": self.scripts if self.scripts else None
        }


def es_query_proto(path, selects, wheres, schema):
    """
    RETURN TEMPLATE AND PATH-TO-FILTER AS A 2-TUPLE
    :param path: THE NESTED PATH (NOT INCLUDING TABLE NAME)
    :param wheres: MAP FROM path TO LIST OF WHERE CONDITIONS
    :return: (es_query, filters_map) TUPLE
    """
    output = None
    last_where = MATCH_ALL
    for p in reversed(sorted( wheres.keys() | set(selects.keys()))):
        where = wheres.get(p)
        select = selects.get(p)

        if where:
            where = AndOp(where).partial_eval().to_esfilter(schema)
            if output:
                where = es_or([es_and([output, where]), where])
        else:
            if output:
                if last_where is MATCH_ALL:
                    where = es_or([output, MATCH_ALL])
                else:
                    where = output
            else:
                where = MATCH_ALL

        if p == ".":
            output = set_default(
                {
                    "from": 0,
                    "size": 0,
                    "sort": [],
                    "query": where
                },
                select.to_es()
            )
        else:
            output = {"nested": {
                "path": p,
                "inner_hits": set_default({"size": 100000}, select.to_es()) if select else None,
                "query": where
            }}

        last_where = where
    return output

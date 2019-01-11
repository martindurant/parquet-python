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

from jx_base.expressions import TupleOp
from jx_base.query import canonical_aggregates
from jx_base.language import is_op
from jx_elasticsearch.es52.aggs import aggs_iterator, count_dim, format_dispatch
from jx_python.containers.cube import Cube
from mo_collections.matrix import Matrix
from mo_dots import Data, coalesce, is_list, set_default, split_field, wrap
from mo_future import sort_using_key
from mo_logs import Log
from mo_logs.strings import quote
from pyLibrary import convert


def format_cube(aggs, es_query, query, decoders, all_selects):
    new_edges = count_dim(aggs, es_query, decoders)

    dims = []
    for e in new_edges:
        if is_op(e.value, TupleOp):
            e.allowNulls = False

        extra = 0 if e.allowNulls is False else 1
        dims.append(len(e.domain.partitions) + extra)

    dims = tuple(dims)
    if any(s.default != canonical_aggregates[s.aggregate].default for s in all_selects):
        # UNUSUAL DEFAULT VALUES MESS THE union() FUNCTION
        is_default = Matrix(dims=dims, zeros=True)
        matricies = {s.name: Matrix(dims=dims) for s in all_selects}
        for row, coord, agg, selects in aggs_iterator(aggs, es_query, decoders):
            for select in selects:
                m = matricies[select.name]
                v = select.pull(agg)
                if v == None:
                    continue
                is_default[coord] = False
                union(m, coord, v, select.aggregate)

        # FILL THE DEFAULT VALUES
        for c, v in is_default:
            if v:
                for s in all_selects:
                    matricies[s.name][c] = s.default
    else:
        matricies = {s.name: Matrix(dims=dims, zeros=s.default) for s in all_selects}
        for row, coord, agg, selects in aggs_iterator(aggs, es_query, decoders):
            for select in selects:
                m = matricies[select.name]
                v = select.pull(agg)
                union(m, coord, v, select.aggregate)

    cube = Cube(
        query.select,
        sort_using_key(new_edges, key=lambda e: e.dim),  # ENSURE EDGES ARE IN SAME ORDER AS QUERY
        matricies
    )
    cube.frum = query
    return cube


def _value_drill(agg):
    while True:
        deeper = agg.get("_nested")
        if deeper:
            agg = deeper
            continue
        deeper = agg.get("_filter")
        if deeper:
            agg = deeper
            continue
        return agg


def format_table(aggs, es_query, query, decoders, all_selects):
    new_edges = wrap(count_dim(aggs, es_query, decoders))
    dims = tuple(len(e.domain.partitions) + (0 if e.allowNulls is False else 1) for e in new_edges)
    rank = len(dims)
    header = tuple(new_edges.name + all_selects.name)
    name2index = {s.name: i + rank for i, s in enumerate(all_selects)}

    def data():
        is_sent = Matrix(dims=dims)
        give_me_zeros = query.sort and not query.groupby
        if give_me_zeros:
            # WE REQUIRE THE ZEROS FOR SORTING
            all_coord = is_sent._all_combos()  # TRACK THE EXPECTED COMBINATIONS
            ordered_coord = all_coord.next()[::-1]
            output = None
            for row, coord, agg, ss in aggs_iterator(aggs, es_query, decoders):
                if coord != ordered_coord:
                    # output HAS BEEN YIELDED, BUT SET THE DEFAULT VALUES
                    if output is not None:
                        for s in all_selects:
                            i = name2index[s.name]
                            if output[i] is None:
                                output[i] = s.default
                        # WE CAN GET THE SAME coord MANY TIMES, SO ONLY ADVANCE WHEN NOT
                        ordered_coord = all_coord.next()[::-1]

                while coord != ordered_coord:
                    # HAPPENS WHEN THE coord IS AHEAD OF ordered_coord
                    record = [d.get_value(ordered_coord[i]) for i, d in enumerate(decoders)] + [s.default for s in all_selects]
                    yield record
                    ordered_coord = all_coord.next()[::-1]
                # coord == missing_coord
                output = [d.get_value(c) for c, d in zip(coord, decoders)] + [None for s in all_selects]
                for select in ss:
                    v = select.pull(agg)
                    if v != None:
                        union(output, name2index[select.name], v, select.aggregate)
                yield output
        else:
            last_coord = None   # HANG ONTO THE output FOR A BIT WHILE WE FILL THE ELEMENTS
            output = None
            for row, coord, agg, ss in aggs_iterator(aggs, es_query, decoders):
                if coord != last_coord:
                    if output:
                        # SET DEFAULTS
                        for i, s in enumerate(all_selects):
                            v = output[rank+i]
                            if v == None:
                                output[rank+i] = s.default
                        yield output
                    output = is_sent[coord]
                    if output == None:
                        output = is_sent[coord] = [d.get_value(c) for c, d in zip(coord, decoders)] + [None for _ in all_selects]
                    last_coord = coord
                # THIS IS A TRICK!  WE WILL UPDATE A ROW THAT WAS ALREADY YIELDED
                for select in ss:
                    v = select.pull(agg)
                    if v != None:
                        union(output, name2index[select.name], v, select.aggregate)

            if output:
                # SET DEFAULTS ON LAST ROW
                for i, s in enumerate(all_selects):
                    v = output[rank+i]
                    if v == None:
                        output[rank+i] = s.default
                yield output

            # EMIT THE MISSING CELLS IN THE CUBE
            if not query.groupby:
                for coord, output in is_sent:
                    if output == None:
                        record = [d.get_value(c) for c, d in zip(coord, decoders)] + [s.default for s in all_selects]
                        yield record

    return Data(
        meta={"format": "table"},
        header=header,
        data=list(data())
    )

def format_tab(aggs, es_query, query, decoders, select):
    table = format_table(aggs, es_query, query, decoders, select)

    def data():
        yield "\t".join(map(quote, table.header))
        for d in table.data:
            yield "\t".join(map(quote, d))

    return data()


def format_csv(aggs, es_query, query, decoders, select):
    table = format_table(aggs, es_query, query, decoders, select)

    def data():
        yield ", ".join(map(quote, table.header))
        for d in table.data:
            yield ", ".join(map(quote, d))

    return data()


def format_list_from_groupby(aggs, es_query, query, decoders, all_selects):
    new_edges = wrap(count_dim(aggs, es_query, decoders))

    def data():
        groupby = query.groupby
        dims = tuple(len(e.domain.partitions) + (0 if e.allowNulls is False else 1) for e in new_edges)
        is_sent = Matrix(dims=dims)
        give_me_zeros = query.sort and not query.groupby

        finishes = []
        # IRREGULAR DEFAULTS MESS WITH union(), SET THEM AT END, IF ANY
        for s in all_selects:
            if s.default != canonical_aggregates[s.aggregate].default:
                s.finish = s.default
                s.default = None
                finishes.append(s)

        for row, coord, agg, _selects in aggs_iterator(aggs, es_query, decoders, give_me_zeros=give_me_zeros):
            output = is_sent[coord]
            if output == None:
                output = is_sent[coord] = Data()
                for g, d, c in zip(groupby, decoders, coord):
                    output[g.put.name] = d.get_value(c)
                for s in all_selects:
                    output[s.name] = s.default
                yield output
            # THIS IS A TRICK!  WE WILL UPDATE A ROW THAT WAS ALREADY YIELDED
            for s in _selects:
                union(output, s.name, s.pull(agg), s.aggregate)

        if finishes:
            # SET ANY DEFAULTS
            for c, o in is_sent:
                for s in finishes:
                    if o[s.name] == None:
                        o[s.name] = s.finish

    for g in query.groupby:
        g.put.name = coalesce(g.put.name, g.name)

    output = Data(
        meta={"format": "list"},
        data=list(data())
    )
    return output


def format_list(aggs, es_query, query, decoders, select):
    table = format_table(aggs, es_query, query, decoders, select)
    header = table.header

    if query.edges or query.groupby:
        data = []
        for row in table.data:
            d = Data()
            for h, r in zip(header, row):
                d[h] = r
            data.append(d)
        format = "list"
    elif is_list(query.select):
        data = Data()
        for h, r in zip(header, table.data[0]):
            data[h] = r
        format = "value"
    else:
        data = table.data[0][0]
        format = "value"

    output = Data(
        meta={"format": format},
        data=data
    )
    return output


def format_line(aggs, es_query, query, decoders, select):
    list = format_list(aggs, es_query, query, decoders, select)

    def data():
        for d in list.data:
            yield convert.value2json(d)

    return data()


set_default(format_dispatch, {
    None: (format_cube, format_table, format_cube, "application/json"),
    "cube": (format_cube, format_cube, format_cube, "application/json"),
    "table": (format_table, format_table, format_table,  "application/json"),
    "list": (format_list, format_list_from_groupby, format_list, "application/json"),
    # "csv": (format_csv, format_csv_from_groupby,  "text/csv"),
    # "tab": (format_tab, format_tab_from_groupby,  "text/tab-separated-values"),
    # "line": (format_line, format_line_from_groupby,  "application/json")
})


def _get(v, k, d):
    for p in split_field(k):
        try:
            v = v.get(p)
            if v is None:
                return d
        except Exception:
            v = [vv.get(p) for vv in v]
    return v


def union(matrix, coord, value, agg):
    # matrix[coord] = existing + value  WITH ADDITIONAL CHECKS
    existing = matrix[coord]
    if existing == None:
        matrix[coord] = value
    elif value == None:
        pass
    elif agg not in ['sum', 'count']:
        if agg == "cardinality" and (existing == 0 or value == 0):
            matrix[coord] = existing + value
            return
        elif agg == "stats" and (not existing or not value):
            matrix[coord] = existing + value
            return
        elif agg == "union":
            matrix[coord] = list(set(existing) | set(value))
            return
        Log.warning("{{agg}} not ready", agg=agg)
    else:
        matrix[coord] = existing + value



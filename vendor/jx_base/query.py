# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from collections import Mapping
from copy import copy

from mo_future import text_type

from jx_base import STRUCT
from jx_base.container import Container
from jx_base.dimensions import Dimension
from jx_base.domains import Domain, SetDomain, DefaultDomain
from jx_base.expressions import jx_expression, Expression, Variable, LeavesOp, ScriptOp, OffsetOp, TRUE, FALSE
from jx_base.queries import is_variable_name
from jx_base.schema import Schema
from mo_dots import Data, relative_field, concat_field
from mo_dots import coalesce, Null, set_default, unwraplist, literal_field
from mo_dots import wrap, unwrap, listwrap
from mo_dots.lists import FlatList
from mo_json.typed_encoder import untype_path
from mo_logs import Log
from mo_math import AND, UNION
from mo_math import Math

DEFAULT_LIMIT = 10
MAX_LIMIT = 10000

_jx = None
_Column = None


def _late_import():
    global _jx
    global _Column

    from jx_python.meta import Column as _Column
    from jx_python import jx as _jx

    _ = _jx
    _ = _Column



class QueryOp(Expression):
    __slots__ = ["frum", "select", "edges", "groupby", "where", "window", "sort", "limit", "having", "format", "isLean"]

    # def __new__(cls, op=None, frum=None, select=None, edges=None, groupby=None, window=None, where=None, sort=None, limit=None, format=None):
    #     output = object.__new__(cls)
    #     for s in QueryOp.__slots__:
    #         setattr(output, s, None)
    #     return output

    def __init__(self, op, frum, select=None, edges=None, groupby=None, window=None, where=None, sort=None, limit=None, format=None):
        if isinstance(frum, Container):
            pass
        else:
            Expression.__init__(self, op, frum)
        self.frum = frum
        self.select = select
        self.edges = edges
        self.groupby = groupby
        self.window = window
        self.where = where
        self.sort = sort
        self.limit = limit
        self.format = format

    def __data__(self):
        def select___data__():
            if isinstance(self.select, list):
                return [s.__data__() for s in self.select]
            else:
                return self.select.__data__()

        return {
            "from": self.frum.__data__(),
            "select": select___data__(),
            "edges": [e.__data__() for e in self.edges],
            "groupby": [g.__data__() for g in self.groupby],
            "window": [w.__data__() for w in self.window],
            "where": self.where.__data__(),
            "sort": self.sort.__data__(),
            "limit": self.limit.__data__()
        }

    def copy(self):
        return QueryOp(
            frum=copy(self.frum),
            select=copy(self.select),
            edges=copy(self.edges),
            groupby=copy(self.groupby),
            window=copy(self.window),
            where=copy(self.where),
            sort=copy(self.sort),
            limit=copy(self.limit),
            format=copy(self.format)
        )


    def vars(self, exclude_where=False, exclude_select=False):
        """
        :return: variables in query
        """
        def edges_get_all_vars(e):
            output = set()
            if isinstance(e.value, text_type):
                output.add(e.value)
            if isinstance(e.value, Expression):
                output |= e.value.vars()
            if e.domain.key:
                output.add(e.domain.key)
            if e.domain.where:
                output |= e.domain.where.vars()
            if e.range:
                output |= e.range.min.vars()
                output |= e.range.max.vars()
            if e.domain.partitions:
                for p in e.domain.partitions:
                    if p.where:
                        output |= p.where.vars()
            return output

        output = set()
        try:
            output |= self.frum.vars()
        except Exception:
            pass

        if not exclude_select:
            for s in listwrap(self.select):
                output |= s.value.vars()
        for s in listwrap(self.edges):
            output |= edges_get_all_vars(s)
        for s in listwrap(self.groupby):
            output |= edges_get_all_vars(s)
        if not exclude_where:
            output |= self.where.vars()
        for s in listwrap(self.sort):
            output |= s.value.vars()

        try:
            output |= UNION(e.vars() for e in self.window)
        except Exception:
            pass

        return output

    def map(self, map_):
        def map_select(s, map_):
            return set_default(
                {"value": s.value.map(map_)},
                s
            )

        def map_edge(e, map_):
            partitions = unwraplist([
                set_default(
                    {"where": p.where.map(map_)},
                    p
                )
                for p in e.domain.partitions
            ])

            domain = copy(e.domain)
            domain.where = e.domain.where.map(map_)
            domain.partitions = partitions

            edge = copy(e)
            edge.value = e.value.map(map_)
            edge.domain = domain
            if e.range:
                edge.range.min = e.range.min.map(map_)
                edge.range.max = e.range.max.map(map_)
            return edge

        if isinstance(self.select, list):
            select = wrap([map_select(s, map_) for s in self.select])
        else:
            select = map_select(self.select, map_)

        return QueryOp(
            "from",
            frum=self.frum.map(map_),
            select=select,
            edges=wrap([map_edge(e, map_) for e in self.edges]),
            groupby=wrap([g.map(map_) for g in self.groupby]),
            window=wrap([w.map(map_) for w in self.window]),
            where=self.where.map(map_),
            sort=wrap([map_select(s, map_) for s in listwrap(self.sort)]),
            limit=self.limit,
            format=self.format
        )

    def missing(self):
        return FALSE

    @staticmethod
    def wrap(query, table=None, schema=None):
        """
        NORMALIZE QUERY SO IT CAN STILL BE JSON
        """
        if isinstance(query, QueryOp) or query == None:
            return query

        query = wrap(query)

        output = QueryOp("from", None)
        output.format = query.format
        output.limit = Math.min(MAX_LIMIT, coalesce(query.limit, DEFAULT_LIMIT))

        # ENSURE from IS A REAL TABLE
        if isinstance(table, Container):
            if table.name != query['from']:
                Log.error("Expecting table to match the 'from' clause")
            output.frum = table
            schema = table.schema
        else:
            from jx_python import wrap_from
            output.frum = wrap_from(query["from"], schema=schema)

        # FIND A SCHEMA IF THERE IS NONE YET
        if not schema and isinstance(output.frum, Schema):
            schema = output.frum
        if not schema and hasattr(output.frum, "schema"):
            schema = output.frum.schema

        if query.select or isinstance(query.select, (Mapping, list)):
            output.select = _normalize_selects(query.select, query.frum, schema=schema)
        else:
            if query.edges or query.groupby:
                output.select = Data(name="count", value=jx_expression("."), aggregate="count", default=0)
            else:
                output.select = _normalize_selects(".", query.frum)

        if query.groupby and query.edges:
            Log.error("You can not use both the `groupby` and `edges` clauses in the same query!")
        elif query.edges:
            output.edges = _normalize_edges(query.edges, limit=output.limit, schema=schema)
            output.groupby = Null
        elif query.groupby:
            output.edges = Null
            output.groupby = _normalize_groupby(query.groupby, limit=output.limit, schema=schema)
        else:
            output.edges = Null
            output.groupby = Null

        output.where = _normalize_where(query.where, schema=schema)
        output.window = [_normalize_window(w) for w in listwrap(query.window)]
        output.having = None
        output.sort = _normalize_sort(query.sort)
        if not Math.is_integer(output.limit) or output.limit < 0:
            Log.error("Expecting limit >= 0")

        output.isLean = query.isLean

        return output


    @property
    def columns(self):
        return listwrap(self.select) + coalesce(self.edges, self.groupby)

    @property
    def query_path(self):
        return "."

    @property
    def column_names(self):
        return listwrap(self.select).name + self.edges.name + self.groupby.name


    def __getitem__(self, item):
        if item == "from":
            return self.frum
        return Data.__getitem__(self, item)

    def copy(self):
        output = object.__new__(QueryOp)
        for s in QueryOp.__slots__:
            setattr(output, s, getattr(self, s))
        return output

    def __data__(self):
        output = wrap({s: getattr(self, s) for s in QueryOp.__slots__})
        return output


canonical_aggregates = wrap({
    "count": {"name": "count", "default": 0},
    "min": {"name": "minimum"},
    "max": {"name": "maximum"},
    "add": {"name": "sum"},
    "avg": {"name": "average"},
    "mean": {"name": "average"},
})


def _normalize_selects(selects, frum, schema=None, ):
    if frum == None or isinstance(frum, (list, set, text_type)):
        if isinstance(selects, list):
            if len(selects) == 0:
                output = Data()
                return output
            else:
                output = [_normalize_select_no_context(s, schema=schema) for s in selects]
        else:
            return _normalize_select_no_context(selects)
    elif isinstance(selects, list):
        output = [ss for s in selects for ss in _normalize_select(s, frum=frum, schema=schema)]
    else:
        output = _normalize_select(selects, frum, schema=schema)

    exists = set()
    for s in output:
        if s.name in exists:
            Log.error("{{name}} has already been defined",  name=s.name)
        exists.add(s.name)
    return output


def _normalize_select(select, frum, schema=None):
    """
    :param select: ONE SELECT COLUMN
    :param frum: TABLE TO get_columns()
    :param schema: SCHEMA TO LOOKUP NAMES FOR DEFINITIONS
    :return: AN ARRAY OF SELECT COLUMNS
    """
    if not _Column:
        _late_import()

    if isinstance(select, text_type):
        canonical = select = Data(value=select)
    else:
        select = wrap(select)
        canonical = select.copy()

    canonical.aggregate = coalesce(canonical_aggregates[select.aggregate].name, select.aggregate, "none")
    canonical.default = coalesce(select.default, canonical_aggregates[canonical.aggregate].default)

    if hasattr(unwrap(frum), "_normalize_select"):
        return frum._normalize_select(canonical)

    output = []
    if not select.value or select.value == ".":
        output.extend([
            set_default(
                {
                    "name": c.name,
                    "value": jx_expression(c.name)
                },
                canonical
            )
            for c in frum.get_leaves()
        ])
    elif isinstance(select.value, text_type):
        if select.value.endswith(".*"):
            canonical.name = coalesce(select.name, ".")
            value = jx_expression(select[:-2])
            if not isinstance(value, Variable):
                Log.error("`*` over general expression not supported yet")
                output.append([
                    set_default(
                        {
                            "value": LeavesOp("leaves", value, prefix=select.prefix),
                            "format": "dict"  # MARKUP FOR DECODING
                        },
                        canonical
                    )
                    for c in frum.get_columns()
                    if c.type not in STRUCT
                ])
            else:
                Log.error("do not know what to do")
        else:
            canonical.name = coalesce(select.name, select.value, select.aggregate)
            canonical.value = jx_expression(select.value)
            output.append(canonical)

    output = wrap(output)
    if any(n==None for n in output.name):
        Log.error("expecting select to have a name: {{select}}", select=select)
    return output


def _normalize_select_no_context(select, schema=None):
    """
    SAME NORMALIZE, BUT NO SOURCE OF COLUMNS
    """
    if not _Column:
        _late_import()

    if isinstance(select, text_type):
        select = Data(value=select)
    else:
        select = wrap(select)

    output = select.copy()
    if not select.value:
        output.name = coalesce(select.name, select.aggregate)
        if output.name:
            output.value = jx_expression(".")
        else:
            return output
    elif isinstance(select.value, text_type):
        if select.value.endswith(".*"):
            name = select.value[:-2]
            output.name = coalesce(select.name,  name)
            output.value = LeavesOp("leaves", Variable(name), prefix=coalesce(select.prefix, name))
        else:
            if select.value == ".":
                output.name = coalesce(select.name, select.aggregate, ".")
                output.value = jx_expression(select.value)
            elif select.value == "*":
                output.name = coalesce(select.name, select.aggregate, ".")
                output.value = LeavesOp("leaves", Variable("."))
            else:
                output.name = coalesce(select.name, select.value, select.aggregate)
                output.value = jx_expression(select.value)
    elif isinstance(select.value, (int, float)):
        if not output.name:
            output.name = text_type(select.value)
        output.value = jx_expression(select.value)
    else:
        output.value = jx_expression(select.value)

    if not output.name:
        Log.error("expecting select to have a name: {{select}}",  select= select)
    if output.name.endswith(".*"):
        Log.error("{{name|quote}} is invalid select", name=output.name)

    output.aggregate = coalesce(canonical_aggregates[select.aggregate].name, select.aggregate, "none")
    output.default = coalesce(select.default, canonical_aggregates[output.aggregate].default)
    return output


def _normalize_edges(edges, limit, schema=None):
    return wrap([n for ie, e in enumerate(listwrap(edges)) for n in _normalize_edge(e, ie, limit=limit, schema=schema)])


def _normalize_edge(edge, dim_index, limit, schema=None):
    """
    :param edge: Not normalized edge
    :param dim_index: Dimensions are ordered; this is this edge's index into that order
    :param schema: for context
    :return: a normalized edge
    """
    if not _Column:
        _late_import()

    if edge == None:
        Log.error("Edge has no value, or expression is empty")
    elif isinstance(edge, text_type):
        if schema:
            leaves = unwraplist(schema.leaves(edge))
            if not leaves or isinstance(leaves, (list, set)):
                return [
                    Data(
                        name=edge,
                        value=jx_expression(edge),
                        allowNulls=True,
                        dim=dim_index
                    )
                ]
            elif isinstance(leaves, _Column):
                return [Data(
                    name=edge,
                    value=jx_expression(edge),
                    allowNulls=True,
                    dim=dim_index,
                    domain=_normalize_domain(domain=leaves, limit=limit, schema=schema)
                )]
            elif isinstance(leaves.fields, list) and len(leaves.fields) == 1:
                return [Data(
                    name=leaves.name,
                    value=jx_expression(leaves.fields[0]),
                    allowNulls=True,
                    dim=dim_index,
                    domain=leaves.getDomain()
                )]
            else:
                return [Data(
                    name=leaves.name,
                    allowNulls=True,
                    dim=dim_index,
                    domain=leaves.getDomain()
                )]
        else:
            return [
                Data(
                    name=edge,
                    value=jx_expression(edge),
                    allowNulls=True,
                    dim=dim_index
                )
            ]
    else:
        edge = wrap(edge)
        if not edge.name and not isinstance(edge.value, text_type):
            Log.error("You must name compound and complex edges: {{edge}}", edge=edge)

        if isinstance(edge.value, (list, set)) and not edge.domain:
            # COMPLEX EDGE IS SHORT HAND
            domain = _normalize_domain(schema=schema)
            domain.dimension = Data(fields=edge.value)

            return [Data(
                name=edge.name,
                value=jx_expression(edge.value),
                allowNulls=bool(coalesce(edge.allowNulls, True)),
                dim=dim_index,
                domain=domain
            )]

        domain = _normalize_domain(edge.domain, schema=schema)

        return [Data(
            name=coalesce(edge.name, edge.value),
            value=jx_expression(edge.value),
            range=_normalize_range(edge.range),
            allowNulls=bool(coalesce(edge.allowNulls, True)),
            dim=dim_index,
            domain=domain
        )]


def _normalize_groupby(groupby, limit, schema=None):
    if groupby == None:
        return None
    output = wrap([n for ie, e in enumerate(listwrap(groupby)) for n in _normalize_group(e, ie, limit, schema=schema) ])
    if any(o==None for o in output):
        Log.error("not expected")
    return output


def _normalize_group(edge, dim_index, limit, schema=None):
    """
    :param edge: Not normalized groupby
    :param dim_index: Dimensions are ordered; this is this groupby's index into that order
    :param schema: for context
    :return: a normalized groupby
    """
    if isinstance(edge, text_type):
        if edge.endswith(".*"):
            prefix = edge[:-2]
            if schema:
                output = wrap([
                    {
                        "name": concat_field(prefix, literal_field(relative_field(untype_path(c.names["."]), prefix))),
                        "put": {"name": literal_field(untype_path(c.names["."]))},
                        "value": jx_expression(c.es_column),
                        "allowNulls": True,
                        "domain": {"type": "default"}
                    }
                    for c in schema.leaves(prefix)
                ])
                return output
            else:
                return wrap([{
                    "name": untype_path(prefix),
                    "put": {"name": literal_field(untype_path(prefix))},
                    "value": jx_expression(prefix),
                    "allowNulls": True,
                    "dim":dim_index,
                    "domain": {"type": "default"}
                }])

        return wrap([{
            "name": edge,
            "value": jx_expression(edge),
            "allowNulls": True,
            "dim":dim_index,
            "domain": {"type": "default"}
        }])
    else:
        edge = wrap(edge)
        if (edge.domain and edge.domain.type != "default") or edge.allowNulls != None:
            Log.error("groupby does not accept complicated domains")

        if not edge.name and not isinstance(edge.value, text_type):
            Log.error("You must name compound edges: {{edge}}",  edge= edge)

        return wrap([{
            "name": coalesce(edge.name, edge.value),
            "value": jx_expression(edge.value),
            "allowNulls": True,
            "dim":dim_index,
            "domain": {"type": "default"}
        }])


def _normalize_domain(domain=None, limit=None, schema=None):
    if not domain:
        return Domain(type="default", limit=limit)
    elif isinstance(domain, _Column):
        if domain.partitions:
            return SetDomain(partitions=domain.partitions.left(limit))
        else:
            return DefaultDomain(type="default", limit=limit)
    elif isinstance(domain, Dimension):
        return domain.getDomain()
    elif schema and isinstance(domain, text_type) and schema[domain]:
        return schema[domain].getDomain()
    elif isinstance(domain, Domain):
        return domain

    if not domain.name:
        domain = domain.copy()
        domain.name = domain.type

    return Domain(**domain)


def _normalize_window(window, schema=None):
    v = window.value
    try:
        expr = jx_expression(v)
    except Exception:
        expr = ScriptOp("script", v)


    return Data(
        name=coalesce(window.name, window.value),
        value=expr,
        edges=[n for e in listwrap(window.edges) for n in _normalize_edge(e, schema)],
        sort=_normalize_sort(window.sort),
        aggregate=window.aggregate,
        range=_normalize_range(window.range),
        where=_normalize_where(window.where, schema=schema)
    )


def _normalize_range(range):
    if range == None:
        return None

    return Data(
        min=None if range.min == None else jx_expression(range.min),
        max=None if range.max == None else jx_expression(range.max),
        mode=range.mode
    )


def _normalize_where(where, schema=None):
    if where == None:
        return TRUE
    return jx_expression(where)


def _map_term_using_schema(master, path, term, schema_edges):
    """
    IF THE WHERE CLAUSE REFERS TO FIELDS IN THE SCHEMA, THEN EXPAND THEM
    """
    output = FlatList()
    for k, v in term.items():
        dimension = schema_edges[k]
        if isinstance(dimension, Dimension):
            domain = dimension.getDomain()
            if dimension.fields:
                if isinstance(dimension.fields, Mapping):
                    # EXPECTING A TUPLE
                    for local_field, es_field in dimension.fields.items():
                        local_value = v[local_field]
                        if local_value == None:
                            output.append({"missing": {"field": es_field}})
                        else:
                            output.append({"term": {es_field: local_value}})
                    continue

                if len(dimension.fields) == 1 and is_variable_name(dimension.fields[0]):
                    # SIMPLE SINGLE-VALUED FIELD
                    if domain.getPartByKey(v) is domain.NULL:
                        output.append({"missing": {"field": dimension.fields[0]}})
                    else:
                        output.append({"term": {dimension.fields[0]: v}})
                    continue

                if AND(is_variable_name(f) for f in dimension.fields):
                    # EXPECTING A TUPLE
                    if not isinstance(v, tuple):
                        Log.error("expecing {{name}}={{value}} to be a tuple",  name= k,  value= v)
                    for i, f in enumerate(dimension.fields):
                        vv = v[i]
                        if vv == None:
                            output.append({"missing": {"field": f}})
                        else:
                            output.append({"term": {f: vv}})
                    continue
            if len(dimension.fields) == 1 and is_variable_name(dimension.fields[0]):
                if domain.getPartByKey(v) is domain.NULL:
                    output.append({"missing": {"field": dimension.fields[0]}})
                else:
                    output.append({"term": {dimension.fields[0]: v}})
                continue
            if domain.partitions:
                part = domain.getPartByKey(v)
                if part is domain.NULL or not part.esfilter:
                    Log.error("not expected to get NULL")
                output.append(part.esfilter)
                continue
            else:
                Log.error("not expected")
        elif isinstance(v, Mapping):
            sub = _map_term_using_schema(master, path + [k], v, schema_edges[k])
            output.append(sub)
            continue

        output.append({"term": {k: v}})
    return {"and": output}


def _where_terms(master, where, schema):
    """
    USE THE SCHEMA TO CONVERT DIMENSION NAMES TO ES FILTERS
    master - TOP LEVEL WHERE (FOR PLACING NESTED FILTERS)
    """
    if isinstance(where, Mapping):
        if where.term:
            # MAP TERM
            try:
                output = _map_term_using_schema(master, [], where.term, schema.edges)
                return output
            except Exception as e:
                Log.error("programmer problem?", e)
        elif where.terms:
            # MAP TERM
            output = FlatList()
            for k, v in where.terms.items():
                if not isinstance(v, (list, set)):
                    Log.error("terms filter expects list of values")
                edge = schema.edges[k]
                if not edge:
                    output.append({"terms": {k: v}})
                else:
                    if isinstance(edge, text_type):
                        # DIRECT FIELD REFERENCE
                        return {"terms": {edge: v}}
                    try:
                        domain = edge.getDomain()
                    except Exception as e:
                        Log.error("programmer error", e)
                    fields = domain.dimension.fields
                    if isinstance(fields, Mapping):
                        or_agg = []
                        for vv in v:
                            and_agg = []
                            for local_field, es_field in fields.items():
                                vvv = vv[local_field]
                                if vvv != None:
                                    and_agg.append({"term": {es_field: vvv}})
                            or_agg.append({"and": and_agg})
                        output.append({"or": or_agg})
                    elif isinstance(fields, list) and len(fields) == 1 and is_variable_name(fields[0]):
                        output.append({"terms": {fields[0]: v}})
                    elif domain.partitions:
                        output.append({"or": [domain.getPartByKey(vv).esfilter for vv in v]})
            return {"and": output}
        elif where["or"]:
            return {"or": [unwrap(_where_terms(master, vv, schema)) for vv in where["or"]]}
        elif where["and"]:
            return {"and": [unwrap(_where_terms(master, vv, schema)) for vv in where["and"]]}
        elif where["not"]:
            return {"not": unwrap(_where_terms(master, where["not"], schema))}
    return where


def _normalize_sort(sort=None):
    """
    CONVERT SORT PARAMETERS TO A NORMAL FORM SO EASIER TO USE
    """

    if sort==None:
        return FlatList.EMPTY

    output = FlatList()
    for s in listwrap(sort):
        if isinstance(s, text_type):
            output.append({"value": jx_expression(s), "sort": 1})
        elif isinstance(s, Expression):
            output.append({"value": s, "sort": 1})
        elif Math.is_integer(s):
            output.append({"value": OffsetOp("offset", s), "sort": 1})
        elif all(d in sort_direction for d in s.values()) and not s.sort and not s.value:
            for v, d in s.items():
                output.append({"value": jx_expression(v), "sort": sort_direction[d]})
        else:
            output.append({"value": jx_expression(coalesce(s.value, s.field)), "sort": coalesce(sort_direction[s.sort], 1)})
    return output


sort_direction = {
    "asc": 1,
    "ascending": 1,
    "desc": -1,
    "descending": -1,
    "none": 0,
    1: 1,
    0: 0,
    -1: -1,
    None: 1,
    Null: 1
}



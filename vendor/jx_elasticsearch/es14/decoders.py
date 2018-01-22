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

from mo_future import text_type, binary_type

from jx_base.dimensions import Dimension
from jx_base.domains import SimpleSetDomain, DefaultDomain, PARTITION
from jx_base.expressions import TupleOp
from jx_base.query import MAX_LIMIT, DEFAULT_LIMIT
from jx_elasticsearch.es14.expressions import Variable, NotOp, InOp, Literal, OrOp, AndOp, InequalityOp, LeavesOp
from jx_python import jx
from mo_dots import set_default, coalesce, literal_field, Data, relative_field
from mo_dots import wrap
from mo_json.typed_encoder import untype_path
from mo_logs import Log
from mo_math import MAX, MIN
from mo_math import Math


class AggsDecoder(object):
    def __new__(cls, e=None, query=None, *args, **kwargs):
        e.allowNulls = coalesce(e.allowNulls, True)

        if e.value and e.domain.type == "default":
            # if query.groupby:
            #     return object.__new__(DefaultDecoder, e)

            if isinstance(e.value, (text_type, binary_type)):
                Log.error("Expecting Variable or Expression, not plain string")

            if isinstance(e.value, LeavesOp):
                return object.__new__(ObjectDecoder, e)
            elif isinstance(e.value, TupleOp):
                # THIS domain IS FROM A dimension THAT IS A SIMPLE LIST OF fields
                # JUST PULL THE FIELDS
                if not all(isinstance(t, Variable) for t in e.value.terms):
                    Log.error("Can only handle variables in tuples")

                e.domain = Data(
                    dimension={"fields": e.value.terms}
                )
                return object.__new__(DimFieldListDecoder, e)
            elif isinstance(e.value, Variable):
                schema = query.frum.schema
                cols = schema.leaves(e.value.var)
                if not cols:
                    return object.__new__(DefaultDecoder, e)
                if len(cols) != 1:
                    return object.__new__(ObjectDecoder, e)
                col = cols[0]
                limit = coalesce(e.domain.limit, query.limit, DEFAULT_LIMIT)

                if col.partitions != None:
                    partitions = col.partitions[:limit:]
                    if e.domain.sort==-1:
                        partitions = list(reversed(sorted(partitions)))
                    else:
                        partitions = sorted(partitions)
                    e.domain = SimpleSetDomain(partitions=partitions, limit=limit)
                else:
                    e.domain = set_default(DefaultDomain(limit=limit), e.domain.__data__())
                    return object.__new__(DefaultDecoder, e)

            else:
                return object.__new__(DefaultDecoder, e)

        if e.value and e.domain.type in PARTITION:
            return object.__new__(SetDecoder, e)
        if isinstance(e.domain.dimension, Dimension):
            e.domain = e.domain.dimension.getDomain()
            return object.__new__(SetDecoder, e)
        if e.value and e.domain.type == "time":
            return object.__new__(TimeDecoder, e)
        if e.range:
            return object.__new__(GeneralRangeDecoder, e)
        if e.value and e.domain.type == "duration":
            return object.__new__(DurationDecoder, e)
        elif e.value and e.domain.type == "range":
            return object.__new__(RangeDecoder, e)
        elif not e.value and e.domain.dimension.fields:
            # THIS domain IS FROM A dimension THAT IS A SIMPLE LIST OF fields
            # JUST PULL THE FIELDS
            fields = e.domain.dimension.fields
            if isinstance(fields, Mapping):
                Log.error("No longer allowed: All objects are expressions")
            else:
                return object.__new__(DimFieldListDecoder, e)
        elif not e.value and all(e.domain.partitions.where):
            return object.__new__(GeneralSetDecoder, e)
        else:
            Log.error("domain type of {{type}} is not supported yet", type=e.domain.type)

    def __init__(self, edge, query, limit):
        self.start = None
        self.edge = edge
        self.name = literal_field(self.edge.name)
        self.query = query
        self.limit = limit
        self.schema = self.query.frum.schema

    def append_query(self, es_query, start):
        Log.error("Not supported")

    def count(self, row):
        pass

    def done_count(self):
        pass

    def get_value_from_row(self, row):
        Log.error("Not implemented")

    def get_value(self, index):
        Log.error("Not implemented")

    def get_index(self, row):
        Log.error("Not implemented")

    @property
    def num_columns(self):
        return 0


class SetDecoder(AggsDecoder):

    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        domain = self.domain = edge.domain

        # WE ASSUME IF THE VARIABLES MATCH, THEN THE SORT TERM AND EDGE TERM MATCH, AND WE SORT BY TERM
        # self.sorted = {1: "asc", -1: "desc", None: None}[getattr(edge.domain, 'sort', None)]
        edge_var = edge.value.vars()
        if query.sort:
            for s in query.sort:
                if not edge_var - s.value.vars():
                    self.sorted = {1: "asc", -1: "desc"}[s.sort]
                    parts = jx.sort(domain.partitions, {"value": domain.key, "sort": s.sort})
                    edge.domain = self.domain = SimpleSetDomain(key=domain.key, label=domain.label, partitions=parts)
        else:
            self.sorted = None

    def append_query(self, es_query, start):
        self.start = start
        domain = self.domain

        domain_key = domain.key
        include, text_include = zip(*(
            (
                float(v) if isinstance(v, (int, float)) else v,
                text_type(float(v)) if isinstance(v, (int, float)) else v
            )
            for v in (p[domain_key] for p in domain.partitions)
        ))
        value = self.edge.value
        exists = AndOp("and", [
            value.exists(),
            InOp("in", [value, Literal("literal", include)])
        ]).partial_eval()

        limit = coalesce(self.limit, len(domain.partitions))

        if isinstance(value, Variable):
            es_field = self.query.frum.schema.leaves(value.var)[0].es_column  # ALREADY CHECKED THERE IS ONLY ONE
            terms = set_default({"terms": {
                "field": es_field,
                "size": limit,
                "order": {"_term": self.sorted} if self.sorted else None
            }}, es_query)
        else:
            terms = set_default({"terms": {
                "script": value.to_ruby(self.schema).script(self.schema),
                "size": limit
            }}, es_query)

        if self.edge.allowNulls:
            missing = set_default(
                {"filter": NotOp("not", exists).to_esfilter(self.schema)},
                es_query
            )
        else:
            missing = None

        return wrap({"aggs": {
            "_match": {
                "filter": exists.to_esfilter(self.schema),
                "aggs": {
                    "_filter": terms
                }
            },
            "_missing": missing
        }})

    def get_value(self, index):
        return self.domain.getKeyByIndex(index)

    def get_value_from_row(self, row):
        return row[self.start].get('key')

    def get_index(self, row):
        try:
            part = row[self.start]
            return self.domain.getIndexByKey(part.get('key'))
        except Exception as e:
            Log.error("problem", cause=e)

    @property
    def num_columns(self):
        return 1


def _range_composer(edge, domain, es_query, to_float, schema):
    # USE RANGES
    _min = coalesce(domain.min, MIN(domain.partitions.min))
    _max = coalesce(domain.max, MAX(domain.partitions.max))

    if edge.allowNulls:
        missing_filter = set_default(
            {
                "filter": NotOp("not", AndOp("and", [
                    edge.value.exists(),
                    InequalityOp("gte", [edge.value, Literal(None, to_float(_min))]),
                    InequalityOp("lt", [edge.value, Literal(None, to_float(_max))])
                ]).partial_eval()).to_esfilter(schema)
            },
            es_query
        )
    else:
        missing_filter = None

    if isinstance(edge.value, Variable):
        calc = {"field": schema.leaves(edge.value.var)[0].es_column}
    else:
        calc = {"script": edge.value.to_ruby(schema).script(schema)}

    return wrap({"aggs": {
        "_match": set_default(
            {"range": calc},
            {"range": {"ranges": [{"from": to_float(p.min), "to": to_float(p.max)} for p in domain.partitions]}},
            es_query
        ),
        "_missing": missing_filter
    }})


class TimeDecoder(AggsDecoder):
    def append_query(self, es_query, start):
        self.start = start
        schema = self.query.frum.schema
        return _range_composer(self.edge, self.edge.domain, es_query, lambda x: x.unix, schema)

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row):
        domain = self.edge.domain
        part = row[self.start]
        if part == None:
            return len(domain.partitions)

        f = coalesce(part.get('from'), part.get('key'))
        t = coalesce(part.get('to'), part.get('key'))
        if f == None or t == None:
            return len(domain.partitions)
        else:
            for p in domain.partitions:
                if p.min.unix <= f < p.max.unix:
                    return p.dataIndex
        sample = part.copy
        sample.buckets = None
        Log.error("Expecting to find {{part}}", part=sample)

    @property
    def num_columns(self):
        return 1


class GeneralRangeDecoder(AggsDecoder):
    """
    Accept an algebraic domain, and an edge with a `range` attribute
    This class assumes the `snapshot` version - where we only include
    partitions that have their `min` value in the range.
    """

    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        if edge.domain.type == "time":
            self.to_float = lambda x: x.unix
        elif edge.domain.type == "range":
            self.to_float = lambda x: x
        else:
            Log.error("Unknown domain of type {{type}} for range edge", type=edge.domain.type)

    def append_query(self, es_query, start):
        self.start = start

        edge = self.edge
        range = edge.range
        domain = edge.domain

        aggs = {}
        for i, p in enumerate(domain.partitions):
            filter_ = AndOp("and", [
                InequalityOp("lte", [range.min, Literal("literal", self.to_float(p.min))]),
                InequalityOp("gt", [range.max, Literal("literal", self.to_float(p.min))])
            ])
            aggs["_join_" + text_type(i)] = set_default(
                {"filter": filter_.to_esfilter(self.schema)},
                es_query
            )

        return wrap({"aggs": aggs})

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row):
        domain = self.edge.domain
        part = row[self.start]
        if part == None:
            return len(domain.partitions)
        return part["_index"]

    @property
    def num_columns(self):
        return 1


class GeneralSetDecoder(AggsDecoder):
    """
    EXPECTING ALL PARTS IN partitions TO HAVE A where CLAUSE
    """

    def append_query(self, es_query, start):
        self.start = start

        parts = self.edge.domain.partitions
        filters = []
        notty = []

        for p in parts:
            w = p.where
            filters.append(AndOp("and", [w] + notty).to_esfilter(self.schema))
            notty.append(NotOp("not", w))

        missing_filter = None
        if self.edge.allowNulls:  # TODO: Use Expression.missing().esfilter() TO GET OPTIMIZED FILTER
            missing_filter = set_default(
                {"filter": AndOp("and", notty).to_esfilter(self.schema)},
                es_query
            )

        return wrap({"aggs": {
            "_match": set_default(
                {"filters": {"filters": filters}},
                es_query
            ),
            "_missing": missing_filter
        }})

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row):
        domain = self.edge.domain
        part = row[self.start]
        # if part == None:
        #     return len(domain.partitions)
        return part.get("_index", len(domain.partitions))

    @property
    def num_columns(self):
        return 1


class DurationDecoder(AggsDecoder):
    def append_query(self, es_query, start):
        self.start = start
        return _range_composer(self.edge, self.edge.domain, es_query, lambda x: x.seconds, self.schema)

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row):
        domain = self.edge.domain
        part = row[self.start]
        if part == None:
            return len(domain.partitions)

        f = coalesce(part.get('from'), part.get('key'))
        t = coalesce(part.get('to'), part.get('key'))
        if f == None or t == None:
            return len(domain.partitions)
        else:
            for p in domain.partitions:
                if p.min.seconds <= f < p.max.seconds:
                    return p.dataIndex
        sample = part.copy
        sample.buckets = None
        Log.error("Expecting to find {{part}}", part=sample)

    @property
    def num_columns(self):
        return 1


class RangeDecoder(AggsDecoder):
    def append_query(self, es_query, start):
        self.start = start
        return _range_composer(self.edge, self.edge.domain, es_query, lambda x: x, self.schema)

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row):
        domain = self.edge.domain
        part = row[self.start]
        if part == None:
            return len(domain.partitions)

        f = coalesce(part.get('from'), part.get('key'))
        t = coalesce(part.get('to'), part.get('key'))
        if f == None or t == None:
            return len(domain.partitions)
        else:
            for p in domain.partitions:
                if p.min <= f < p.max:
                    return p.dataIndex
        sample = part.copy
        sample.buckets = None
        Log.error("Expecting to find {{part}}", part=sample)

    @property
    def num_columns(self):
        return 1


class ObjectDecoder(AggsDecoder):
    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        if isinstance(edge.value, LeavesOp):
            prefix = edge.value.term.var
            flatter = lambda k: literal_field(relative_field(k, prefix))
        else:
            prefix = edge.value.var
            flatter = lambda k: relative_field(k, prefix)

        self.put, self.fields = zip(*[
            (flatter(untype_path(c.names["."])), c.es_column)
            for c in query.frum.schema.leaves(prefix)
        ])

        self.domain = self.edge.domain = wrap({"dimension": {"fields": self.fields}})
        self.domain.limit = Math.min(coalesce(self.domain.limit, query.limit, 10), MAX_LIMIT)
        self.parts = list()
        self.key2index = {}
        self.computed_domain = False

    def append_query(self, es_query, start):
        self.start = start
        for i, v in enumerate(self.fields):
            nest = wrap({"aggs": {
                "_match": set_default({"terms": {
                    "field": v,
                    "size": self.domain.limit
                }}, es_query),
                "_missing": set_default(
                    {"filter": {"missing": {"field": v}}},
                    es_query
                )
            }})
            es_query = nest
        return es_query

    def count(self, row):
        value = self.get_value_from_row(row)
        i = self.key2index.get(value)
        if i is None:
            i = self.key2index[value] = len(self.parts)
            self.parts.append(value)

    def done_count(self):
        self.computed_domain = True
        self.edge.domain = self.domain = SimpleSetDomain(
            key="value",
            partitions=[{"value": p, "dataIndex": i} for i, p in enumerate(self.parts)]
        )

    def get_index(self, row):
        value = self.get_value_from_row(row)
        if self.computed_domain:
            return self.domain.getIndexByKey(value)

        if value is None:
            return -1
        i = self.key2index.get(value)
        if i is None:
            i = self.key2index[value] = len(self.parts)
            self.parts.append(value)
        return i

    def get_value_from_row(self, row):
        part = row[self.start:self.start + self.num_columns:]
        if not part[0]['doc_count']:
            return None

        output = Data()
        for k, v in zip(self.put, part):
            output[k] = v.get('key')
        return output

    @property
    def num_columns(self):
        return len(self.fields)


class DefaultDecoder(SetDecoder):
    # FOR DECODING THE default DOMAIN TYPE (UNKNOWN-AT-QUERY-TIME SET OF VALUES)

    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        self.domain = edge.domain
        self.domain.limit = Math.min(coalesce(self.domain.limit, query.limit, 10), MAX_LIMIT)
        self.parts = list()
        self.key2index = {}
        self.computed_domain = False

        # WE ASSUME IF THE VARIABLES MATCH, THEN THE SORT TERM AND EDGE TERM MATCH, AND WE SORT BY TERM
        self.sorted = None
        edge_var = edge.value.vars()
        for s in query.sort:
            if not edge_var - s.value.vars():
                self.sorted = {1: "asc", -1: "desc"}[s.sort]

    def append_query(self, es_query, start):
        self.start = start

        value = self.edge.value.partial_eval()
        script = value.to_ruby(self.schema)
        exists = NotOp("not", script.miss).partial_eval()
        if not isinstance(self.edge.value, Variable):

            output = wrap({"aggs": {
                "_match": {
                    "filter": exists.to_esfilter(self.schema),
                    "aggs": {
                        "_filter": set_default(
                            {"terms": {
                                "script": script.expr,
                                "size": self.domain.limit,
                                "order": {"_term": self.sorted} if self.sorted else None
                            }},
                            es_query
                        )
                    }
                },
                "_missing": set_default(
                    {"filter": NotOp("not", exists).to_esfilter(self.schema)},
                    es_query
                )
            }})
            return output
        elif self.edge.value.var in [s.value.var for s in self.query.sort]:
            sort_dir = [s.sort for s in self.query.sort if s.value.var == self.edge.value.var][0]
            output = wrap({"aggs": {
                "_match": set_default(
                    {"terms": {
                        "field": self.schema.leaves(self.edge.value.var)[0].es_column,
                        "size": self.domain.limit,
                        "order": {"_term": "asc" if sort_dir == 1 else "desc"}
                    }},
                    es_query
                ),
                "_missing": set_default(
                    {"filter": NotOp("not", exists).to_esfilter(self.schema)},
                    es_query
                )
            }})
            return output
        else:
            output = wrap({"aggs": {
                "_match": set_default(
                    {"terms": {
                        "field": self.schema.leaves(self.edge.value.var)[0].es_column,
                        "size": self.domain.limit
                    }},
                    es_query
                ),
                "_missing": set_default(
                    {"filter": NotOp("not", exists).to_esfilter(self.schema)},
                    es_query
                )
            }})
            return output

    def count(self, row):
        part = row[self.start]
        if part['doc_count']:
            if part.get('key') != None:
                self.parts.append(part.get('key'))
            else:
                self.edge.allowNulls = True  # OK! WE WILL ALLOW NULLS

    def done_count(self):
        self.edge.domain = self.domain = SimpleSetDomain(
            partitions=jx.sort(set(self.parts))
        )
        self.parts = None
        self.computed_domain = True

    def get_index(self, row):
        if self.computed_domain:
            try:
                part = row[self.start]
                return self.domain.getIndexByKey(part.get('key'))
            except Exception as e:
                Log.error("problem", cause=e)
        else:
            try:
                part = row[self.start]
                key = part.get('key')
                i = self.key2index.get(key)
                if i is None:
                    i = len(self.parts)
                    part = {"key": key, "dataIndex": i}
                    self.parts.append({"key": key, "dataIndex": i})
                    self.key2index[i] = part
                return i
            except Exception as e:
                Log.error("problem", cause=e)

    @property
    def num_columns(self):
        return 1


class DimFieldListDecoder(SetDecoder):
    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        self.fields = edge.domain.dimension.fields
        self.domain = self.edge.domain
        self.domain.limit = Math.min(coalesce(self.domain.limit, query.limit, 10), MAX_LIMIT)
        self.parts = list()

    def append_query(self, es_query, start):
        # TODO: USE "reverse_nested" QUERY TO PULL THESE
        self.start = start
        for i, v in enumerate(self.fields):
            exists = v.exists().partial_eval()
            nest = wrap({"aggs": {"_match": {
                "filter": exists.to_esfilter(self.schema),
                "aggs": {"_filter": set_default({"terms": {
                    "field": self.schema.leaves(v.var)[0].es_column,
                    "size": self.domain.limit
                }}, es_query)}
            }}})
            if self.edge.allowNulls:
                nest.aggs._missing = set_default(
                    {"filter": NotOp("not", exists).to_esfilter(self.schema)},
                    es_query
                )
            es_query = nest

        if self.domain.where:
            filter_ = self.domain.where.partial_eval().to_esfilter(self.schema)
            es_query = {"aggs": {"_filter": set_default({"filter": filter_}, es_query)}}

        return es_query

    def count(self, row):
        part = row[self.start:self.start + len(self.fields):]
        if part[0]['doc_count']:
            value = tuple(p.get("key") for p in part)
            self.parts.append(value)

    def done_count(self):
        columns = map(text_type, range(len(self.fields)))
        parts = wrap([{text_type(i): p for i, p in enumerate(part)} for part in set(self.parts)])
        self.parts = None
        sorted_parts = jx.sort(parts, columns)

        self.edge.domain = self.domain = SimpleSetDomain(
            key="value",
            partitions=[{"value": tuple(v[k] for k in columns), "dataIndex": i} for i, v in enumerate(sorted_parts)]
        )

    def get_index(self, row):
        find = tuple(p.get("key") for p in row[self.start:self.start + self.num_columns:])
        return self.domain.getIndexByKey(find)

    @property
    def num_columns(self):
        return len(self.fields)



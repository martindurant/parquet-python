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
from jx_base.dimensions import Dimension
from jx_base.domains import DefaultDomain, PARTITION, SimpleSetDomain
from jx_base.expressions import ExistsOp, FirstOp, GtOp, GteOp, LeavesOp, LtOp, LteOp, MissingOp, TupleOp, Variable
from jx_base.query import DEFAULT_LIMIT, MAX_LIMIT
from jx_base.language import is_op
from jx_elasticsearch.es52.es_query import Aggs, FilterAggs, FiltersAggs, NestedAggs, RangeAggs, TermsAggs
from jx_elasticsearch.es52.expressions import AndOp, InOp, Literal, NotOp
from jx_elasticsearch.es52.painless import LIST_TO_PIPE, Painless
from jx_elasticsearch.es52.util import pull_functions
from jx_python import jx
from mo_dots import Data, coalesce, concat_field, is_data, literal_field, relative_field, set_default, wrap
from mo_future import first, text_type, transpose
from mo_json import EXISTS, OBJECT, STRING
from mo_json.typed_encoder import EXISTS_TYPE, NESTED_TYPE, untype_path
from mo_logs import Log
from mo_logs.strings import expand_template, quote
import mo_math
from mo_math import MAX, MIN

DEBUG = False


class AggsDecoder(object):
    def __new__(cls, e=None, query=None, *args, **kwargs):
        e.allowNulls = coalesce(e.allowNulls, True)

        if e.value and e.domain.type == "default":
            # if query.groupby:
            #     return object.__new__(DefaultDecoder, e)

            if is_text(e.value):
                Log.error("Expecting Variable or Expression, not plain string")

            if is_op(e.value, LeavesOp):
                return object.__new__(ObjectDecoder)
            elif is_op(e.value, TupleOp):
                # THIS domain IS FROM A dimension THAT IS A SIMPLE LIST OF fields
                # JUST PULL THE FIELDS
                if not all(is_op(t, Variable) for t in e.value.terms):
                    Log.error("Can only handle variables in tuples")

                e.domain = Data(
                    dimension={"fields": e.value.terms}
                )
                return object.__new__(DimFieldListDecoder)

            elif is_op(e.value, Variable):
                schema = query.frum.schema
                cols = schema.leaves(e.value.var)
                if not cols:
                    return object.__new__(DefaultDecoder)
                if len(cols) != 1:
                    return object.__new__(ObjectDecoder)
                col = first(cols)
                limit = coalesce(e.domain.limit, query.limit, DEFAULT_LIMIT)

                if col.cardinality == None:
                    DEBUG and Log.warning(
                        "metadata for column {{name|quote}} (id={{id}}) is not ready",
                        name=concat_field(col.es_index, col.es_column),
                        id=id(col)
                    )
                    e.domain = set_default(DefaultDomain(limit=limit), e.domain.__data__())
                    return object.__new__(DefaultDecoder)
                elif col.partitions == None:
                    e.domain = set_default(DefaultDomain(limit=limit), e.domain.__data__())
                    return object.__new__(DefaultDecoder)
                else:
                    DEBUG and Log.note("id={{id}} has parts!!!", id=id(col))
                    if col.multi > 1 and len(col.partitions) < 10:
                        return object.__new__(MultivalueDecoder)

                    partitions = col.partitions[:limit:]
                    if e.domain.sort == -1:
                        partitions = list(reversed(sorted(partitions)))
                    else:
                        partitions = sorted(partitions)
                    e.domain = SimpleSetDomain(partitions=partitions, limit=limit)

            else:
                return object.__new__(DefaultDecoder)

        if e.value and e.domain.type in PARTITION:
            return object.__new__(SetDecoder)
        if isinstance(e.domain.dimension, Dimension):
            e.domain = e.domain.dimension.getDomain()
            return object.__new__(SetDecoder)
        if e.value and e.domain.type == "time":
            return object.__new__(TimeDecoder)
        if e.range:
            return object.__new__(GeneralRangeDecoder)
        if e.value and e.domain.type == "duration":
            return object.__new__(DurationDecoder)
        elif e.value and e.domain.type == "range":
            return object.__new__(RangeDecoder)
        elif not e.value and e.domain.dimension.fields:
            # THIS domain IS FROM A dimension THAT IS A SIMPLE LIST OF fields
            # JUST PULL THE FIELDS
            fields = e.domain.dimension.fields
            if is_data(fields):
                Log.error("No longer allowed: All objects are expressions")
            else:
                return object.__new__(DimFieldListDecoder)
        elif not e.value and all(e.domain.partitions.where):
            return object.__new__(GeneralSetDecoder)
        else:
            Log.error("domain type of {{type}} is not supported yet", type=e.domain.type)

    def __init__(self, edge, query, limit):
        self.edge = edge
        self.name = literal_field(self.edge.name)
        self.query = query
        self.limit = limit
        self.schema = self.query.frum.schema

    def append_query(self, query_path, es_query):
        Log.error("Not supported")

    def count(self, row):
        pass

    # DO NOT IMPLEMENT IF domain HAS KNOWN PARTITIONS
    # def done_count(self):
    #     pass

    def get_value_from_row(self, row):
        raise NotImplementedError()

    def get_value(self, index):
        raise NotImplementedError()

    def get_index(self, row, es_query=None, index=None):
        raise NotImplementedError()

    @property
    def num_columns(self):
        return 0


class SetDecoder(AggsDecoder):

    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        domain = self.domain = edge.domain
        self.sorted = None
        self.pull = pull_functions[STRING]

        # WE ASSUME IF THE VARIABLES MATCH, THEN THE SORT TERM AND EDGE TERM MATCH, AND WE SORT BY TERM
        # self.sorted = {1: "asc", -1: "desc", None: None}[getattr(edge.domain, 'sort', None)]
        edge_var = set(v.var for v in edge.value.vars())
        if query.sort:
            for s in query.sort:
                if not edge_var - set(v.var for v in s.value.vars()):
                    self.sorted = {1: "asc", -1: "desc"}[s.sort]
                    parts = jx.sort(domain.partitions, {"value": domain.key, "sort": s.sort})
                    edge.domain = self.domain = SimpleSetDomain(key=domain.key, label=domain.label, partitions=parts)

    def append_query(self, query_path, es_query):
        domain = self.domain
        domain_key = domain.key
        value = Painless[self.edge.value]
        cnv = pull_functions[value.type]
        include = tuple(cnv(p[domain_key]) for p in domain.partitions)

        exists = Painless[AndOp([
            InOp([value, Literal(include)])
        ])].partial_eval()

        limit = coalesce(self.limit, len(domain.partitions))

        if is_op(value, Variable):
            es_field = first(self.query.frum.schema.leaves(value.var)).es_column  # ALREADY CHECKED THERE IS ONLY ONE
            match = TermsAggs(
                "_match",
                {
                    "field": es_field,
                    "size": limit,
                    "order": {"_term": self.sorted} if self.sorted else None
                },
                self
            )
        else:
            match = TermsAggs(
                "_match",
                {
                    "script": text_type(value.to_es_script(self.schema)),
                    "size": limit
                },
                self
            )
        output = Aggs().add(FilterAggs("_filter", exists, None).add(match.add(es_query)))

        if self.edge.allowNulls:
            # FIND NULLS AT EACH NESTED LEVEL
            for p in self.schema.query_path:
                if p == query_path:
                    # MISSING AT THE QUERY DEPTH
                    output.add(
                        NestedAggs(p).add(FilterAggs("_missing0", NotOp(exists), self).add(es_query))
                    )
                else:
                    # PARENT HAS NO CHILDREN, SO MISSING
                    column = first(self.schema.values(query_path, (OBJECT, EXISTS)))
                    output.add(
                        NestedAggs(column.nested_path[0]).add(
                            FilterAggs(
                                "_missing1",
                                NotOp(ExistsOp(Variable(column.es_column.replace(NESTED_TYPE, EXISTS_TYPE)))),
                                self
                            ).add(es_query)
                        )
                    )
        return output

    def get_value(self, index):
        return self.domain.getKeyByIndex(index)

    def get_value_from_row(self, parts):
        key = parts[0].get('key')
        return self.pull(key)

    def get_index(self, row, es_query=None, index=None):
        try:
            key = row[0].get('key')
            return self.domain.getIndexByKey(key)
        except Exception as e:
            Log.error("problem", cause=e)

    @property
    def num_columns(self):
        return 1


def _range_composer(self, edge, domain, es_query, to_float, schema):
    # USE RANGES
    _min = coalesce(domain.min, MIN(domain.partitions.min))
    _max = coalesce(domain.max, MAX(domain.partitions.max))

    output = Aggs()
    if edge.allowNulls:
        output.add(FilterAggs(
            "_missing",
            NotOp(AndOp([
                edge.value.exists(),
                GteOp([edge.value, Literal(to_float(_min))]),
                LtOp([edge.value, Literal(to_float(_max))])
            ]).partial_eval()),
            self
        ).add(es_query))

    if is_op(edge.value, Variable):
        calc = {"field": first(schema.leaves(edge.value.var)).es_column}
    else:
        calc = {"script": text_type(Painless[edge.value].to_es_script(schema))}
    calc['ranges'] = [{"from": to_float(p.min), "to": to_float(p.max)} for p in domain.partitions]

    return output.add(RangeAggs("_match", calc, self).add(es_query))


class TimeDecoder(AggsDecoder):
    def append_query(self, query_path, es_query):
        schema = self.query.frum.schema
        return _range_composer(self, self.edge, self.edge.domain, es_query, lambda x: x.unix, schema)

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row, es_query=None, index=None):
        domain = self.edge.domain
        part = row[0]
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

    def append_query(self, query_path, es_query):
        edge = self.edge
        range = edge.range
        domain = edge.domain

        aggs = Aggs()
        for i, p in enumerate(domain.partitions):
            filter_ = AndOp([
                LteOp([range.min, Literal(self.to_float(p.min))]),
                GtOp([range.max, Literal(self.to_float(p.min))])
            ])
            aggs.add(FilterAggs("_match" + text_type(i), filter_, self).add(es_query))

        return aggs

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row, es_query=None, index=None):
        domain = self.edge.domain
        part = row[0]
        if part == None:
            return len(domain.partitions)
        index = int(es_query.name[6:])
        return index

    @property
    def num_columns(self):
        return 1


class GeneralSetDecoder(AggsDecoder):
    """
    EXPECTING ALL PARTS IN partitions TO HAVE A where CLAUSE
    """

    def append_query(self, query_path, es_query):
        parts = self.edge.domain.partitions
        filters = []
        notty = []
        for p in parts:
            w = p.where
            filters.append(AndOp([w] + notty))
            notty.append(NotOp(w))

        output = Aggs().add(FiltersAggs("_match", filters, self).add(es_query))
        if self.edge.allowNulls:  # TODO: Use Expression.missing().esfilter() TO GET OPTIMIZED FILTER
            output.add(FilterAggs("_missing", AndOp(notty), self).add(es_query))

        return output

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row, es_query=None, index=None):
        domain = self.edge.domain
        if index == None:
            return len(domain.partitions)
        else:
            return index

    @property
    def num_columns(self):
        return 1


class DurationDecoder(AggsDecoder):
    def append_query(self, query_path, es_query):
        return _range_composer(self, self.edge, self.edge.domain, es_query, lambda x: x.seconds, self.schema)

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row, es_query=None, index=None):
        domain = self.edge.domain
        part = row[0]
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
    def append_query(self, query_path, es_query):
        return _range_composer(self, self.edge, self.edge.domain, es_query, lambda x: x, self.schema)

    def get_value(self, index):
        return self.edge.domain.getKeyByIndex(index)

    def get_index(self, row, es_query=None, index=None):
        domain = self.edge.domain
        part = row[0]
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


class MultivalueDecoder(SetDecoder):
    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        self.var = edge.value.var
        self.values = query.frum.schema[edge.value.var][0].partitions
        self.parts = []

    def append_query(self, query_path, es_query):
        es_field = first(self.query.frum.schema.leaves(self.var)).es_column

        return Aggs().add(TermsAggs("_match", {
            "script": expand_template(LIST_TO_PIPE, {"expr": 'doc[' + quote(es_field) + '].values'})
        }, self).add(es_query))

    def get_value_from_row(self, row):
        values = row[0]['key'].replace("||", "\b").split("|")
        if len(values) == 2:
            return None
        t = tuple(v.replace("\b", "|") for v in sorted(values[1:-1]))

        if len(t) == 0:
            return None
        elif len(t) == 1:
            return t[0]
        else:
            return t

    def get_index(self, row, es_query=None, index=None):
        find = self.get_value_from_row(row)
        return self.domain.getIndexByKey(find)

    def count(self, row):
        value = self.get_value_from_row(row)
        self.parts.append(value)

    def done_count(self):
        self.edge.allowNulls = False
        self.edge.domain = self.domain = SimpleSetDomain(
            partitions=jx.sort(set(self.parts))
        )
        self.parts = None

    @property
    def num_columns(self):
        return 1


class ObjectDecoder(AggsDecoder):
    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        if is_op(edge.value, LeavesOp):
            prefix = edge.value.term.var
            flatter = lambda k: literal_field(relative_field(k, prefix))
        else:
            prefix = edge.value.var
            flatter = lambda k: relative_field(k, prefix)

        self.put, self.fields = transpose(*[
            (flatter(untype_path(c.name)), c.es_column)
            for c in query.frum.schema.leaves(prefix)
        ])

        self.domain = self.edge.domain = wrap({"dimension": {"fields": self.fields}})
        self.domain.limit = mo_math.min(coalesce(self.domain.limit, query.limit, 10), MAX_LIMIT)
        self.parts = list()
        self.key2index = {}
        self.computed_domain = False

    def append_query(self, query_path, es_query):
        decoder = self
        for i, v in enumerate(self.fields):
            nest = Aggs().add(
                TermsAggs("_match", {
                    "field": v,
                    "size": self.domain.limit
                }, decoder).add(es_query)
            ).add(
                FilterAggs("_missing", MissingOp(Variable(v)), decoder).add(es_query)
            )
            es_query = nest
            decoder = None
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

    def get_index(self, row, es_query=None, index=None):
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

    def get_value_from_row(self, parts):
        if not parts[0]['doc_count']:
            return None

        output = Data()
        for k, v in transpose(self.put, parts):
            output[k] = v.get('key')
        return output

    def get_value(self, index):
        return self.parts[index]

    @property
    def num_columns(self):
        return len(self.fields)


class DefaultDecoder(SetDecoder):
    # FOR DECODING THE default DOMAIN TYPE (UNKNOWN-AT-QUERY-TIME SET OF VALUES)

    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        self.domain = edge.domain
        self.domain.limit = mo_math.min(coalesce(self.domain.limit, query.limit, 10), MAX_LIMIT)
        self.parts = list()
        self.key2index = {}
        self.computed_domain = False
        self.script = Painless[self.edge.value].partial_eval().to_es_script(self.schema)
        self.pull = pull_functions[self.script.data_type]
        self.missing = self.script.miss.partial_eval()
        self.exists = NotOp(self.missing).partial_eval()

        # WHEN SORT VALUE AND EDGE VALUE MATCHES, WE SORT BY TERM
        sort_candidates = [s for s in query.sort if s.value == edge.value]
        if sort_candidates:
            self.es_order = {"_term": {1: "asc", -1: "desc"}[sort_candidates[0].sort]}
        else:
            self.es_order = None

    def append_query(self, query_path, es_query):
        if is_op(self.edge.value, FirstOp) and is_op(self.edge.value.term, Variable):
            self.edge.value = self.edge.value.term  # ES USES THE FIRST TERM FOR {"terms": } AGGREGATION
        if not is_op(self.edge.value, Variable):
            terms = TermsAggs(
                "_match",
                {
                    "script": {"lang": "painless", "inline": self.script.expr},
                    "size": self.domain.limit,
                    "order": self.es_order
                },
                self
            )
        else:
            terms = TermsAggs(
                "_match", {
                    "field": first(self.schema.leaves(self.edge.value.var)).es_column,
                    "size": self.domain.limit,
                    "order": self.es_order
                },
                self
            )
        output = Aggs()
        output.add(FilterAggs("_filter", self.exists, None).add(terms.add(es_query)))
        output.add(FilterAggs("_missing", self.missing, self).add(es_query))
        return output

    def count(self, row):
        part = row[0]
        if part['doc_count']:
            key = part.get('key')
            if key != None:
                self.parts.append(self.pull(key))
            else:
                self.edge.allowNulls = True  # OK! WE WILL ALLOW NULLS

    def done_count(self):
        self.edge.domain = self.domain = SimpleSetDomain(
            partitions=jx.sort(set(self.parts))
        )
        self.parts = None
        self.computed_domain = True

    def get_index(self, row, es_query=None, index=None):
        if self.computed_domain:
            try:
                part = row[0]
                return self.domain.getIndexByKey(self.pull(part.get('key')))
            except Exception as e:
                Log.error("problem", cause=e)
        else:
            try:
                part = row[0]
                key = self.pull(part.get('key'))
                i = self.key2index.get(key)
                if i is None:
                    i = len(self.parts)
                    part = {"key": key, "dataIndex": i}
                    self.parts.append(part)
                    self.key2index[key] = i
                return i
            except Exception as e:
                Log.error("problem", cause=e)

    @property
    def num_columns(self):
        return 1


class DimFieldListDecoder(SetDecoder):
    def __init__(self, edge, query, limit):
        AggsDecoder.__init__(self, edge, query, limit)
        edge.allowNulls = False
        self.fields = edge.domain.dimension.fields
        self.domain = self.edge.domain
        self.domain.limit = mo_math.min(coalesce(self.domain.limit, query.limit, 10), MAX_LIMIT)
        self.parts = list()

    def append_query(self, query_path, es_query):
        decoder = self
        for i, v in enumerate(self.fields):
            exists = v.exists().partial_eval()
            nest = Aggs()
            nest.add(TermsAggs("_match", {
                "field": first(self.schema.leaves(v.var)).es_column,
                "size": self.domain.limit
            }, decoder).add(es_query))
            nest.add(FilterAggs("_missing", NotOp(exists), decoder).add(es_query))
            es_query = nest
            decoder = None

        if self.domain.where:
            es_query = FilterAggs("_filter", self.domain.where, None).add(es_query)

        return es_query

    def count(self, parts):
        if parts[0]['doc_count']:
            value = tuple(p.get("key") for p, f in zip(parts, self.fields))
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

    def get_index(self, row, es_query=None, index=None):
        if row[0]['doc_count']:
            find = tuple(p.get("key") for p, f in zip(row, self.fields))
            output = self.domain.getIndexByKey(find)
            return output

    @property
    def num_columns(self):
        return len(self.fields)



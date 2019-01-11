# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
from __future__ import absolute_import, division, unicode_literals

from jx_elasticsearch.es52.expressions import ES52
from mo_dots import is_data, is_list, startswith_field
from mo_future import text_type
from mo_json import value2json
from mo_logs import Log

_new = object.__new__


class Aggs(object):

    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.decoders = []
        self.selects = []

    def to_es(self, schema, query_path="."):
        if self.children:
            return {"aggs": {
                name: t.to_es(schema, query_path)
                for i, t in enumerate(self.children)
                for name in [t.name if t.name else "_" + text_type(i)]
            }}
        else:
            return {}

    def add(self, child):
        self.children.append(child)
        return self

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, Aggs) and self.name == other.name

    def merge(self, other):
        if self != other:
            return False
        self.children.extend(other.children)
        self.decoders.extend(other.decoders)
        return True

    def __str__(self):
        return value2json(self.to_es)

    def copy(self):
        output = _new(self.__class__)
        output.name = self.name
        output.children = self.children[:]
        output.decoders = self.decoders[:]
        output.selects = self.selects[:]
        return output


class ExprAggs(Aggs):

    def __init__(self, name, expr, select):
        Aggs.__init__(self, name)
        self.expr = expr
        self.selects = [select]

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, ExprAggs) and self.name == other.name and self.expr == other.expr

    def merge(self, other):
        if self != other:
            return False
        self.expr += other.expr
        self.children.extend(other.children)
        self.decoders.extend(other.decoders)
        self.selects.extend(other.selects)
        return True

    def to_es(self, schema, query_path="."):
        self.expr['aggs'] = Aggs.to_es(self, schema, query_path).get('aggs')
        return self.expr

    def copy(self):
        output = Aggs.copy(self)
        output.expr = self.expr
        return output


class FilterAggs(Aggs):
    def __init__(self, name, filter, decoder):
        Aggs.__init__(self, name)
        self.filter = filter
        if is_data(filter):
            Log.error("programming error")
        self.decoders = [decoder] if decoder else []

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, FilterAggs) and self.name == other.name and self.filter == other.filter

    def merge(self, other):
        if self != other:
            return False
        self.children.extend(other.children)
        self.decoders.extend(other.decoders)
        return True

    def to_es(self, schema, query_path="."):
        output = Aggs.to_es(self, schema, query_path)
        output['filter'] = ES52[self.filter].partial_eval().to_esfilter(schema)
        return output

    def copy(self):
        output = Aggs.copy(self)
        output.filter = self.filter
        return output


class ComplexAggs(FilterAggs):
    """
    FOR COMPLICATED AGGREGATIONS
    """

    def __init__(self, select):
        Aggs.__init__(self, "_filter")
        self.expr = {"filter": {"match_all": {}}}
        self.selects = [select]

    def to_es(self, schema, query_path="."):
        self.expr['aggs'] = Aggs.to_es(self, schema, query_path).get('aggs')
        return self.expr

    def copy(self):
        output = Aggs.copy(self)
        output.expr = self.expr
        return output


class FiltersAggs(Aggs):
    def __init__(self, name, filters, decoder):
        Aggs.__init__(self, name)
        self.filters = filters
        self.decoders = [decoder] if decoder else []
        if not is_list(filters):
            Log.error("expecting a list")

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, FiltersAggs) and self.name == other.name and self.filters == other.filters

    def merge(self, other):
        if self != other:
            return False
        self.children.extend(other.children)
        self.decoders.extend(other.decoders)
        return True

    def to_es(self, schema, query_path="."):
        output = Aggs.to_es(self, schema, query_path)
        output['filters'] = {"filters": [f.partial_eval().to_esfilter(schema) for f in self.filters]}
        return output

    def copy(self):
        output = Aggs.copy(self)
        output.filters = self.filters
        return output


class NestedAggs(Aggs):
    def __init__(self, path):
        Aggs.__init__(self, "_nested")
        self.path = path

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, NestedAggs) and self.path == other.path

    def to_es(self, schema, query_path="."):
        output = Aggs.to_es(self, schema, self.path)
        if query_path == self.path:
            Log.error("this should have been cancelled out")
        elif startswith_field(self.path, query_path):
            output['nested'] = {"path": self.path}
        else:
            output["reverse_nested"] = {"path": None if self.path == "." else self.path}
        return output

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, NestedAggs) and self.path == other.path

    def copy(self):
        output = Aggs.copy(self)
        output.path = self.path
        return output


class TermsAggs(Aggs):
    def __init__(self, name, terms, decoder):
        Aggs.__init__(self, name)
        self.terms = terms
        self.decoders = [decoder] if decoder else []

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, TermsAggs) and self.name == other.name and self.terms == other.terms

    def to_es(self, schema, query_path="."):
        output = Aggs.to_es(self, schema, query_path)
        output['terms'] = self.terms
        return output

    def copy(self):
        output = Aggs.copy(self)
        output.terms = self.terms
        return output


class RangeAggs(Aggs):
    def __init__(self, name, expr, decoder):
        Aggs.__init__(self, name)
        self.expr = expr
        self.decoders = [decoder] if decoder else []

    def __eq__(self, other):
        if self is other:
            return True
        return isinstance(other, RangeAggs) and self.name == other.name and self.expr == other.expr

    def to_es(self, schema, query_path="."):
        output = Aggs.to_es(self, schema, query_path)
        output['range'] = self.expr
        return output

    def copy(self):
        output = Aggs.copy(self)
        output.expr = self.expr
        return output


def simplify(aggs):
    # CONVERT FROM TREE TO UNION OF SEQUENCES
    def depth_first(aggr):
        if aggr.__class__ == Aggs:
            # BASE CLASS Aggs IS ONLY A PLACEHOLDER
            if not aggr.children:
                yield tuple()
                return
            for c in aggr.children:
                for path in depth_first(c):
                    yield path
        elif not aggr.children:
            yield (aggr,)
        else:
            for c in aggr.children:
                for path in depth_first(c):
                    yield (aggr,) + path

    # CANCEL OUT REDUNDANT NESTED AGGS
    combined = []
    for path in depth_first(aggs):
        current_nested = NestedAggs(".")
        prev = None
        remove = []
        for step in path:
            if isinstance(step, NestedAggs):
                if prev is not None:
                    remove.append(prev)
                    prev = None
                if current_nested is not None:
                    if current_nested.path == step.path:
                        remove.append(step)
                        continue
                    else:
                        pass
                prev = step
            else:
                current_nested = prev if prev else current_nested
                prev = None

        combined.append(tuple(p for p in path if not any(p is r for r in remove)))

    # COMMON FACTOR, CONVERT BACK TO TREE
    def merge(aggregations):
        output = []
        while True:
            common_children = []
            first_found = None
            common = None
            for i, terms in enumerate(aggregations):
                if not terms:
                    continue
                term, rest = terms[0], terms[1:]
                if first_found is None:
                    first_found = term
                    common_children.append(rest)
                    common = first_found.copy()
                    aggregations[i] = None
                elif term == first_found:
                    common_children.append(rest)
                    common.selects.extend([t for t in term.selects if not any(t is s for s in common.selects)])
                    common.decoders.extend([t for t in term.decoders if not any(t is d for d in common.decoders)])
                    aggregations[i] = None

            if first_found is None:
                return output
            else:
                common.children = merge(common_children)
            output.append(common)

    merged = [trim_root(o) for o in merge(combined)]

    output = Aggs()
    output.children = merged
    return output


def trim_root(agg):
    if isinstance(agg, NestedAggs) and agg.path == '.':
        if len(agg.children) == 1:
            return agg.children[0]
        else:
            output = Aggs()
            output.children = agg.children
            return output
    else:
        return agg

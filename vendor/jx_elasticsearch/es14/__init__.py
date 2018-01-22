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

from jx_base import container
from jx_base.container import Container
from jx_base.dimensions import Dimension
from jx_base.expressions import jx_expression
from jx_base.queries import is_variable_name
from jx_base.query import QueryOp
from jx_base.schema import Schema
from jx_elasticsearch.es14.aggs import es_aggsop, is_aggsop
from jx_elasticsearch.es14.deep import is_deepop, es_deepop
from jx_elasticsearch.es14.setop import is_setop, es_setop
from jx_elasticsearch.es14.util import aggregates
from jx_elasticsearch.meta import FromESMetadata
from jx_python import jx
from mo_dots import Data, Null, unwrap
from mo_dots import coalesce, split_field, literal_field, unwraplist, join_field
from mo_dots import wrap, listwrap
from mo_dots.lists import FlatList
from mo_json import scrub
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import Except
from pyLibrary import convert
from pyLibrary.env import elasticsearch, http


class ES14(Container):
    """
    SEND jx QUERIES TO ElasticSearch
    """

    def __new__(cls, *args, **kwargs):
        if (len(args) == 1 and args[0].get("index") == "meta") or kwargs.get("index") == "meta":
            output = FromESMetadata.__new__(FromESMetadata, *args, **kwargs)
            output.__init__(*args, **kwargs)
            return output
        else:
            return Container.__new__(cls)

    @override
    def __init__(
        self,
        host,
        index,
        type=None,
        alias=None,
        name=None,
        port=9200,
        read_only=True,
        timeout=None,  # NUMBER OF SECONDS TO WAIT FOR RESPONSE, OR SECONDS TO WAIT FOR DOWNLOAD (PASSED TO requests)
        wait_for_active_shards=1,  # ES WRITE CONSISTENCY (https://www.elastic.co/guide/en/elasticsearch/reference/1.7/docs-index_.html#index-consistency)
        typed=None,
        kwargs=None
    ):
        Container.__init__(self, None)
        if not container.config.default:
            container.config.default = {
                "type": "elasticsearch",
                "settings": unwrap(kwargs)
            }
        self.settings = kwargs
        self.name = coalesce(name, alias, index)
        if read_only:
            self._es = elasticsearch.Alias(alias=coalesce(alias, index), kwargs=kwargs)
        else:
            self._es = elasticsearch.Cluster(kwargs=kwargs).get_index(read_only=read_only, kwargs=kwargs)

        self.meta = FromESMetadata(kwargs=kwargs)
        self.settings.type = self._es.settings.type
        self.edges = Data()
        self.worker = None

        columns = self.meta.get_columns(table_name=coalesce(name, alias, index))
        self._schema = Schema(coalesce(name, alias, index), columns)

        if typed == None:
            # SWITCH ON TYPED MODE
            self.typed = any(c.es_column.find(".$") != -1 for c in columns)
        else:
            self.typed = typed

    @property
    def schema(self):
        return self._schema

    def __data__(self):
        settings = self.settings.copy()
        settings.settings = None
        return settings

    def __enter__(self):
        Log.error("No longer used")
        return self

    def __exit__(self, type, value, traceback):
        if not self.worker:
            return

        if isinstance(value, Exception):
            self.worker.stop()
            self.worker.join()
        else:
            self.worker.join()

    @property
    def query_path(self):
        return join_field(split_field(self.name)[1:])

    @property
    def url(self):
        return self._es.url

    def query(self, _query):
        try:
            query = QueryOp.wrap(_query, schema=self)

            for n in self.namespaces:
                query = n.convert(query)

            for s in listwrap(query.select):
                if not aggregates.get(s.aggregate):
                    Log.error(
                        "ES can not aggregate {{name}} because {{aggregate|quote}} is not a recognized aggregate",
                        name=s.name,
                        aggregate=s.aggregate
                    )

            frum = query["from"]
            if isinstance(frum, QueryOp):
                result = self.query(frum)
                q2 = query.copy()
                q2.frum = result
                return jx.run(q2)

            if is_deepop(self._es, query):
                return es_deepop(self._es, query)
            if is_aggsop(self._es, query):
                return es_aggsop(self._es, frum, query)
            if is_setop(self._es, query):
                return es_setop(self._es, query)
            Log.error("Can not handle")
        except Exception as e:
            e = Except.wrap(e)
            if "Data too large, data for" in e:
                http.post(self._es.cluster.path+"/_cache/clear")
                Log.error("Problem (Tried to clear Elasticsearch cache)", e)
            Log.error("problem", e)

    def addDimension(self, dim):
        if isinstance(dim, list):
            Log.error("Expecting dimension to be a object, not a list:\n{{dim}}",  dim= dim)
        self._addDimension(dim, [])

    def _addDimension(self, dim, path):
        dim.full_name = dim.name
        for e in dim.edges:
            d = Dimension(e, dim, self)
            self.edges[d.full_name] = d

    def __getitem__(self, item):
        c = self.get_columns(table_name=self.name, column_name=item)
        if c:
            if len(c) > 1:
                Log.error("Do not know how to handle multipole matches")
            return c[0]

        e = self.edges[item]
        if not c:
            Log.warning("Column with name {{column|quote}} can not be found in {{table}}", column=item, table=self.name)
        return e

    def __getattr__(self, item):
        return self.edges[item]

    def update(self, command):
        """
        EXPECTING command == {"set":term, "where":where}
        THE set CLAUSE IS A DICT MAPPING NAMES TO VALUES
        THE where CLAUSE IS AN ES FILTER
        """
        command = wrap(command)
        schema = self._es.get_properties()

        # GET IDS OF DOCUMENTS
        results = self._es.search({
            "fields": listwrap(schema._routing.path),
            "query": {"filtered": {
                "filter": jx_expression(command.where).to_esfilter(Null)
            }},
            "size": 10000
        })

        # SCRIPT IS SAME FOR ALL (CAN ONLY HANDLE ASSIGNMENT TO CONSTANT)
        scripts = FlatList()
        for k, v in command.set.items():
            if not is_variable_name(k):
                Log.error("Only support simple paths for now")
            if isinstance(v, Mapping) and v.doc:
                scripts.append({"doc": v.doc})
            else:
                v = scrub(v)
                scripts.append({"script": "ctx._source." + k + " = " + jx_expression(v).to_ruby(schema).script(schema)})

        if results.hits.hits:
            updates = []
            for h in results.hits.hits:
                for s in scripts:
                    updates.append({"update": {"_id": h._id, "_routing": unwraplist(h.fields[literal_field(schema._routing.path)])}})
                    updates.append(s)
            content = ("\n".join(convert.value2json(c) for c in updates) + "\n").encode('utf-8')
            response = self._es.cluster.post(
                self._es.path + "/_bulk",
                data=content,
                headers={"Content-Type": "application/json"},
                timeout=self.settings.timeout,
                params={"wait_for_active_shards": self.settings.wait_for_active_shards}
            )
            if response.errors:
                Log.error("could not update: {{error}}", error=[e.error for i in response["items"] for e in i.values() if e.status not in (200, 201)])


# encoding: utf-8
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

import re
from collections import Mapping
from copy import deepcopy

import mo_json
from jx_python import jx
from jx_python.expressions import jx_expression_to_function
from jx_python.meta import Column
from mo_dots import coalesce, Null, Data, set_default, listwrap, literal_field, ROOT_PATH, concat_field, split_field
from mo_dots import wrap
from mo_dots.lists import FlatList
from mo_future import text_type, binary_type
from mo_json import value2json
from mo_json.typed_encoder import EXISTS_TYPE, BOOLEAN_TYPE, STRING_TYPE, NUMBER_TYPE, NESTED_TYPE, TYPE_PREFIX
from mo_kwargs import override
from mo_logs import Log, strings
from mo_logs.exceptions import Except
from mo_logs.strings import utf82unicode
from mo_math import Math
from mo_math.randoms import Random
from mo_threads import Lock
from mo_threads import ThreadedQueue
from mo_threads import Till
from mo_times.dates import Date
from mo_times.timer import Timer
from pyLibrary import convert
from pyLibrary.env import http

ES_STRUCT = ["object", "nested"]
ES_NUMERIC_TYPES = ["long", "integer", "double", "float"]
ES_PRIMITIVE_TYPES = ["string", "boolean", "integer", "date", "long", "double"]
INDEX_DATE_FORMAT = "%Y%m%d_%H%M%S"


class Features(object):
    pass


class Index(Features):
    """
    AN ElasticSearch INDEX LIFETIME MANAGEMENT TOOL

    ElasticSearch'S REST INTERFACE WORKS WELL WITH PYTHON AND JAVASCRIPT
    SO HARDLY ANY LIBRARY IS REQUIRED.  IT IS SIMPLER TO MAKE HTTP CALLS
    DIRECTLY TO ES USING YOUR FAVORITE HTTP LIBRARY.  I HAVE SOME
    CONVENIENCE FUNCTIONS HERE, BUT IT'S BETTER TO MAKE YOUR OWN.

    THIS CLASS IS TO HELP DURING ETL, CREATING INDEXES, MANAGING ALIASES
    AND REMOVING INDEXES WHEN THEY HAVE BEEN REPLACED.  IT USES A STANDARD
    SUFFIX (YYYYMMDD-HHMMSS) TO TRACK AGE AND RELATIONSHIP TO THE ALIAS,
    IF ANY YET.

    """
    @override
    def __init__(
        self,
        index,  # NAME OF THE INDEX, EITHER ALIAS NAME OR FULL VERSION NAME
        id_column="_id",
        type=None,  # SCHEMA NAME, (DEFAULT TO TYPE IN INDEX, IF ONLY ONE)
        alias=None,
        explore_metadata=True,  # PROBING THE CLUSTER FOR METADATA IS ALLOWED
        read_only=True,
        tjson=None,  # STORED AS TYPED JSON
        timeout=None,  # NUMBER OF SECONDS TO WAIT FOR RESPONSE, OR SECONDS TO WAIT FOR DOWNLOAD (PASSED TO requests)
        consistency="one",  # ES WRITE CONSISTENCY (https://www.elastic.co/guide/en/elasticsearch/reference/1.7/docs-index_.html#index-consistency)
        debug=False,  # DO NOT SHOW THE DEBUG STATEMENTS
        cluster=None,
        kwargs=None
    ):
        if index==None:
            Log.error("not allowed")
        # if index == alias:
        #     Log.error("must have a unique index name")

        self.cluster_state = None
        self.debug = debug
        self.settings = kwargs
        if cluster:
            self.cluster = cluster
        else:
            self.cluster = Cluster(kwargs)

        try:
            full_index = self.get_index(index)
            if full_index and alias==None:
                kwargs.alias = kwargs.index
                kwargs.index = full_index
            if full_index==None:
                Log.error("not allowed")
            if type == None:
                # NO type PROVIDED, MAYBE THERE IS A SUITABLE DEFAULT?
                with self.cluster.metadata_locker:
                    index_ = self.cluster._metadata.indices[self.settings.index]
                if not index_:
                    indices = self.cluster.get_metadata().indices
                    index_ = indices[self.settings.index]

                candidate_types = list(index_.mappings.keys())
                if len(candidate_types) != 1:
                    Log.error("Expecting `type` parameter")
                self.settings.type = type = candidate_types[0]
        except Exception as e:
            # EXPLORING (get_metadata()) IS NOT ALLOWED ON THE PUBLIC CLUSTER
            Log.error("not expected", cause=e)

        if not type:
            Log.error("not allowed")

        self.path = "/" + full_index + "/" + type

        if self.debug:
            Log.alert("elasticsearch debugging for {{url}} is on", url=self.url)

        if tjson:
            from pyLibrary.env.typed_inserter import TypedInserter

            self.encode = TypedInserter(self, id_column).typed_encode
        else:
            if tjson is None and not read_only:
                kwargs.tjson = False
                Log.warning("{{index}} is not typed", index=self.settings.index)
            self.encode = get_encoder(id_column)

    @property
    def url(self):
        return self.cluster.path.rstrip("/") + "/" + self.path.lstrip("/")

    def get_properties(self, retry=True):
        if self.settings.explore_metadata:
            metadata = self.cluster.get_metadata()
            index = metadata.indices[self.settings.index]

            if index == None and retry:
                #TRY AGAIN, JUST IN CASE
                self.cluster.cluster_state = None
                return self.get_properties(retry=False)

            if not index.mappings[self.settings.type]:
                Log.warning(
                    "ElasticSearch index {{index|quote}} does not have type {{type|quote}} in {{metadata|json}}",
                    index=self.settings.index,
                    type=self.settings.type,
                    metadata=jx.sort(metadata.indices.keys())
                )
                return Null
            return index.mappings[self.settings.type].properties
        else:
            mapping = self.cluster.get(self.path + "/_mapping")
            if not mapping[self.settings.type]:
                Log.error(
                    "ElasticSearch index {{index|quote}} does not have type {{type|quote}}",
                    index=self.settings.index,
                    type=self.settings.type
                )
            return wrap({"mappings": mapping[self.settings.type]})

    def delete_all_but_self(self):
        """
        DELETE ALL INDEXES WITH GIVEN PREFIX, EXCEPT name
        """
        prefix = self.settings.alias
        name = self.settings.index

        if prefix == name:
            Log.note("{{index_name}} will not be deleted",  index_name= prefix)
        for a in self.cluster.get_aliases():
            # MATCH <prefix>YYMMDD_HHMMSS FORMAT
            if re.match(re.escape(prefix) + "\\d{8}_\\d{6}", a.index) and a.index != name:
                self.cluster.delete_index(a.index)

    def add_alias(self, alias=None):
        alias = coalesce(alias, self.settings.alias)
        self.cluster_state = None
        self.cluster.post(
            "/_aliases",
            data={
                "actions": [
                    {"add": {"index": self.settings.index, "alias": alias}}
                ]
            },
            timeout=coalesce(self.settings.timeout, 30)
        )
        self.settings.alias = alias

        # WAIT FOR ALIAS TO APPEAR
        while True:
            response = self.cluster.get("/_cluster/state", retry={"times": 5}, timeout=3)
            if alias in response.metadata.indices[self.settings.index].aliases:
                return
            Log.note("Waiting for alias {{alias}} to appear", alias=alias)
            Till(seconds=1).wait()



    def get_index(self, alias):
        """
        RETURN THE INDEX USED BY THIS alias
        """
        alias_list = self.cluster.get_aliases()
        output = jx.sort(set([
            a.index
            for a in alias_list
            if a.alias == alias or
                a.index == alias or
                (re.match(re.escape(alias) + "\\d{8}_\\d{6}", a.index) and a.index != alias)
        ]))

        if len(output) > 1:
            Log.error("only one index with given alias==\"{{alias}}\" expected",  alias= alias)

        if not output:
            return Null

        return output.last()

    def is_proto(self, index):
        """
        RETURN True IF THIS INDEX HAS NOT BEEN ASSIGNED ITS ALIAS
        """
        for a in self.cluster.get_aliases():
            if a.index == index and a.alias:
                return False
        return True

    def flush(self, forced=False):
        try:
            self.cluster.post("/" + self.settings.index + "/_flush", data={"wait_if_ongoing": True, "forced": forced})
        except Exception as e:
            if "FlushNotAllowedEngineException" in e:
                Log.note("Flush is ignored")
            else:
                Log.error("Problem flushing", cause=e)

    def refresh(self):
        self.cluster.post("/" + self.settings.index + "/_refresh")

    def delete_record(self, filter):
        if self.settings.read_only:
            Log.error("Index opened in read only mode, no changes allowed")
        self.cluster.get_metadata()

        if self.cluster.cluster_state.version.number.startswith("0.90"):
            query = {"filtered": {
                "query": {"match_all": {}},
                "filter": filter
            }}
        elif self.cluster.cluster_state.version.number.startswith("1."):
            query = {"query": {"filtered": {
                "query": {"match_all": {}},
                "filter": filter
            }}}
        else:
            raise NotImplementedError

        if self.debug:
            Log.note("Delete bugs:\n{{query}}",  query= query)

        result = self.cluster.delete(
            self.path + "/_query",
            data=value2json(query),
            timeout=600,
            params={"consistency": self.settings.consistency}
        )

        for name, status in result._indices.items():
            if status._shards.failed > 0:
                Log.error("Failure to delete from {{index}}", index=name)


    def extend(self, records):
        """
        records - MUST HAVE FORM OF
            [{"value":value}, ... {"value":value}] OR
            [{"json":json}, ... {"json":json}]
            OPTIONAL "id" PROPERTY IS ALSO ACCEPTED
        """
        if self.settings.read_only:
            Log.error("Index opened in read only mode, no changes allowed")
        lines = []
        try:
            for r in records:
                rec = self.encode(r)
                json_bytes = rec['json'].encode('utf8')
                lines.append(b'{"index":{"_id": ' + convert.value2json(rec['id']).encode("utf8") + b'}}')
                lines.append(json_bytes)

            del records

            if not lines:
                return

            with Timer("Add {{num}} documents to {{index}}", {"num": len(lines) / 2, "index":self.settings.index}, debug=self.debug):
                try:
                    data_bytes = b"\n".join(l for l in lines) + b"\n"
                except Exception as e:
                    raise Log.error("can not make request body from\n{{lines|indent}}", lines=lines, cause=e)

                wait_for_active_shards = coalesce(
                    self.settings.wait_for_active_shards,
                    {"one": 1, None: None}[self.settings.consistency]
                )

                response = self.cluster.post(
                    self.path + "/_bulk",
                    data=data_bytes,
                    headers={"Content-Type": "application/x-ndjson"},
                    timeout=self.settings.timeout,
                    retry=self.settings.retry,
                    params={"wait_for_active_shards": wait_for_active_shards}
                )
                items = response["items"]

                fails = []
                if self.cluster.version.startswith("0.90."):
                    for i, item in enumerate(items):
                        if not item.index.ok:
                            fails.append(i)
                elif self.cluster.version.startswith(("1.4.", "1.5.", "1.6.", "1.7.", "5.", "6.")):
                    for i, item in enumerate(items):
                        if item.index.status not in [200, 201]:
                            fails.append(i)
                else:
                    Log.error("version not supported {{version}}", version=self.cluster.version)

                if fails:
                    if len(fails) <= 3:
                        cause = [
                            Except(
                                template="{{status}} {{error}} (and {{some}} others) while loading line id={{id}} into index {{index|quote}} (typed={{tjson}}):\n{{line}}",
                                status=items[i].index.status,
                                error=items[i].index.error,
                                some=len(fails) - 1,
                                line=strings.limit(lines[i * 2 + 1].decode('utf8'), 500 if not self.debug else 100000),
                                index=self.settings.index,
                                tjson=self.settings.tjson,
                                id=items[i].index._id
                            )
                            for i in fails
                        ]
                    else:
                        i=fails[0]
                        cause = Except(
                            template="{{status}} {{error}} (and {{some}} others) while loading line id={{id}} into index {{index|quote}} (typed={{tjson}}):\n{{line}}",
                            status=items[i].index.status,
                            error=items[i].index.error,
                            some=len(fails) - 1,
                            line=strings.limit(lines[i * 2 + 1].decode('utf8'), 500 if not self.debug else 100000),
                            index=self.settings.index,
                            tjson=self.settings.tjson,
                            id=items[i].index._id
                        )
                    Log.error("Problems with insert", cause=cause)

        except Exception as e:
            if e.message.startswith("sequence item "):
                Log.error("problem with {{data}}", data=text_type(repr(lines[int(e.message[14:16].strip())])), cause=e)
            Log.error("problem sending to ES", e)

    # RECORDS MUST HAVE id AND json AS A STRING OR
    # HAVE id AND value AS AN OBJECT
    def add(self, record):
        if self.settings.read_only:
            Log.error("Index opened in read only mode, no changes allowed")
        if isinstance(record, list):
            Log.error("add() has changed to only accept one record, no lists")
        self.extend([record])

    def refresh(self):
        self.cluster.post("/" + self.settings.index + "/_refresh")

    def set_refresh_interval(self, seconds, **kwargs):
        """
        :param seconds:  -1 FOR NO REFRESH
        :param kwargs: ANY OTHER REQUEST PARAMETERS
        :return: None
        """
        if seconds <= 0:
            interval = -1
        else:
            interval = text_type(seconds) + "s"

        if self.cluster.version.startswith("0.90."):
            response = self.cluster.put(
                "/" + self.settings.index + "/_settings",
                data='{"index":{"refresh_interval":' + value2json(interval) + '}}',
                **kwargs
            )

            result = mo_json.json2value(utf82unicode(response.all_content))
            if not result.ok:
                Log.error("Can not set refresh interval ({{error}})", {
                    "error": utf82unicode(response.all_content)
                })
        elif self.cluster.version.startswith(("1.4.", "1.5.", "1.6.", "1.7.", "5.", "6.")):
            result = self.cluster.put(
                "/" + self.settings.index + "/_settings",
                data=convert.unicode2utf8('{"index":{"refresh_interval":' + value2json(interval) + '}}'),
                **kwargs
            )

            if not result.acknowledged:
                Log.error("Can not set refresh interval ({{error}})", {
                    "error": result
                })
        else:
            Log.error("Do not know how to handle ES version {{version}}", version=self.cluster.version)

    def search(self, query, timeout=None, retry=None):
        query = wrap(query)
        try:
            if self.debug:
                if len(query.facets.keys()) > 20:
                    show_query = query.copy()
                    show_query.facets = {k: "..." for k in query.facets.keys()}
                else:
                    show_query = query
                Log.note("Query:\n{{query|indent}}", query=show_query)
            return self.cluster.post(
                self.path + "/_search",
                data=query,
                timeout=coalesce(timeout, self.settings.timeout),
                retry=retry
            )
        except Exception as e:
            Log.error(
                "Problem with search (path={{path}}):\n{{query|indent}}",
                path=self.path + "/_search",
                query=query,
                cause=e
            )

    def threaded_queue(self, batch_size=None, max_size=None, period=None, silent=False):

        def errors(e, _buffer):  # HANDLE ERRORS FROM extend()
            if e.cause.cause:
                not_possible = [f for f in listwrap(e.cause.cause) if any(h in f for h in HOPELESS)]
                still_have_hope = [f for f in listwrap(e.cause.cause) if all(h not in f for h in HOPELESS)]
            else:
                not_possible = [e]
                still_have_hope = []

            if still_have_hope:
                if "429 EsRejectedExecutionException[rejected execution (queue capacity" in e:
                    Log.note("waiting for ES to be free ({{num}} pending)", num=len(_buffer))
                elif "503 UnavailableShardsException" in e:
                    Log.note("waiting for ES to initialize shards ({{num}} pending)", num=len(_buffer))
                else:
                    Log.warning("Problem with sending to ES ({{num}} pending)", num=len(_buffer), cause=still_have_hope)
            elif not_possible:
                # THERE IS NOTHING WE CAN DO
                Log.warning("Not inserted, will not try again", cause=not_possible[0:10:])
                del _buffer[:]

        return ThreadedQueue(
            "push to elasticsearch: " + self.settings.index,
            self,
            batch_size=batch_size,
            max_size=max_size,
            period=period,
            silent=silent,
            error_target=errors
        )

    def delete(self):
        self.cluster.delete_index(index_name=self.settings.index)


HOPELESS = [
    "Document contains at least one immense term",
    "400 MapperParsingException",
    "400 RoutingMissingException",
    "500 IllegalArgumentException[cannot change DocValues type",
    "JsonParseException"
]



known_clusters = {}

class Cluster(object):

    @override
    def __new__(cls, host, port=9200, kwargs=None):
        if not isinstance(port, int):
            Log.error("port must be integer")
        cluster = known_clusters.get((host, port))
        if cluster:
            return cluster

        cluster = object.__new__(cls)
        known_clusters[(host, port)] = cluster
        return cluster

    @override
    def __init__(self, host, port=9200, explore_metadata=True, kwargs=None):
        """
        settings.explore_metadata == True - IF PROBING THE CLUSTER FOR METADATA IS ALLOWED
        settings.timeout == NUMBER OF SECONDS TO WAIT FOR RESPONSE, OR SECONDS TO WAIT FOR DOWNLOAD (PASSED TO requests)
        """
        if hasattr(self, "settings"):
            return

        self.settings = kwargs
        self.cluster_state = None
        self._metadata = None
        self.metadata_locker = Lock()
        self.debug = kwargs.debug
        self.version = None
        self.path = kwargs.host + ":" + text_type(kwargs.port)
        self.get_metadata()

    @override
    def get_or_create_index(
        self,
        index,
        alias=None,
        schema=None,
        limit_replicas=None,
        read_only=False,
        tjson=None,
        kwargs=None
    ):
        best = self._get_best(kwargs)
        if not best:
            output = self.create_index(kwargs=kwargs, schema=schema, limit_replicas=limit_replicas)
            return output
        elif best.alias != None:
            kwargs.alias = best.alias
            kwargs.index = best.index
        elif kwargs.alias == None:
            kwargs.alias = best.alias
            kwargs.index = best.index

        index = kwargs.index
        meta = self.get_metadata()
        columns = parse_properties(index, ".", meta.indices[index].mappings.values()[0].properties)

        tjson = kwargs.tjson
        if len(columns) != 0:
            kwargs.tjson = tjson or any(
                c.names["."].startswith(TYPE_PREFIX) or
                c.names["."].find("." + TYPE_PREFIX) != -1
                for c in columns
            )
        if tjson is None and not kwargs.tjson:
            Log.warning("Not typed index, columns are:\n{{columns|json}}", columns=columns)

        return Index(kwargs=kwargs, cluster=self)

    def _get_best(self, settings):
        aliases = self.get_aliases()
        indexes = jx.sort([
            a
            for a in aliases
            if (a.alias == settings.index and settings.alias == None) or
            (re.match(re.escape(settings.index) + r'\d{8}_\d{6}', a.index) and settings.alias == None) or
            (a.index == settings.index and (settings.alias == None or a.alias == None or a.alias == settings.alias))
        ], "index")
        return indexes.last()

    @override
    def get_index(self, index, type=None, alias=None, tjson=None, read_only=True, kwargs=None):
        """
        TESTS THAT THE INDEX EXISTS BEFORE RETURNING A HANDLE
        """
        if read_only:
            # GET EXACT MATCH, OR ALIAS
            aliases = self.get_aliases()
            if index in aliases.index:
                pass
            elif index in aliases.alias:
                match = [a for a in aliases if a.alias == index][0]
                kwargs.alias = match.alias
                kwargs.index = match.index
            else:
                Log.error("Can not find index {{index_name}}", index_name=kwargs.index)
            return Index(kwargs=kwargs, cluster=self)
        else:
            # GET BEST MATCH, INCLUDING PROTOTYPE
            best = self._get_best(kwargs)
            if not best:
                Log.error("Can not find index {{index_name}}", index_name=kwargs.index)

            if best.alias != None:
                kwargs.alias = best.alias
                kwargs.index = best.index
            elif kwargs.alias == None:
                kwargs.alias = kwargs.index
                kwargs.index = best.index

            if tjson is None:
                metadata = self.get_metadata()
                metadata[kwargs.index]

            return Index(kwargs=kwargs, cluster=self)

    def get_alias(self, alias):
        """
        RETURN REFERENCE TO ALIAS (MANY INDEXES)
        USER MUST BE SURE NOT TO SEND UPDATES
        """
        aliases = self.get_aliases()
        if alias in aliases.alias:
            settings = self.settings.copy()
            settings.alias = alias
            settings.index = alias
            return Index(read_only=True, kwargs=settings, cluster=self)
        Log.error("Can not find any index with alias {{alias_name}}",  alias_name= alias)

    def get_prototype(self, alias):
        """
        RETURN ALL INDEXES THAT ARE INTENDED TO BE GIVEN alias, BUT HAVE NO
        ALIAS YET BECAUSE INCOMPLETE
        """
        output = sort([
            a.index
            for a in self.get_aliases()
            if re.match(re.escape(alias) + "\\d{8}_\\d{6}", a.index) and not a.alias
        ])
        return output

    @override
    def create_index(
        self,
        index,
        alias=None,
        create_timestamp=None,
        schema=None,
        limit_replicas=None,
        limit_replicas_warning=True,
        read_only=False,
        tjson=True,
        kwargs=None
    ):
        if not alias:
            alias = kwargs.alias = kwargs.index
            index = kwargs.index = proto_name(alias, create_timestamp)

        if kwargs.alias == index:
            Log.error("Expecting index name to conform to pattern")

        if kwargs.schema_file:
            Log.error('schema_file attribute not supported.  Use {"$ref":<filename>} instead')

        if schema == None:
            Log.error("Expecting a schema")
        elif isinstance(schema, text_type):
            Log.error("Expecting a schema")
        elif self.version.startswith("5."):
            schema.settings.index.max_inner_result_window = None  # NOT ACCEPTED BY ES5
            schema = mo_json.json2value(value2json(schema), leaves=True)
        elif self.version.startswith("6."):
            schema = mo_json.json2value(value2json(schema), leaves=True)
        else:
            schema = retro_schema(mo_json.json2value(value2json(schema), leaves=True))

        for m in schema.mappings.values():
            if tjson:
                m.properties[EXISTS_TYPE] = {"type": "long", "store": True}
            m.dynamic_templates = DEFAULT_DYNAMIC_TEMPLATES + m.dynamic_templates + [{
                "default_all": {
                    "mapping": {"store": True},
                    "match": "*"
                }
            }]

        if limit_replicas:
            # DO NOT ASK FOR TOO MANY REPLICAS
            health = self.get("/_cluster/health")
            if schema.settings.index.number_of_replicas >= health.number_of_nodes:
                if limit_replicas_warning:
                    Log.warning(
                        "Reduced number of replicas: {{from}} requested, {{to}} realized",
                        {"from": schema.settings.index.number_of_replicas},
                        to=health.number_of_nodes - 1
                    )
                schema.settings.index.number_of_replicas = health.number_of_nodes - 1

        self.put(
            "/" + index,
            data=schema,
            headers={"Content-Type": "application/json"}
        )

        # CONFIRM INDEX EXISTS
        while True:
            try:
                state = self.get("/_cluster/state", retry={"times": 5}, timeout=3)
                if index in state.metadata.indices:
                    self._metadata = None
                    break
                Log.note("Waiting for index {{index}} to appear", index=index)
            except Exception as e:
                Log.warning("Problem while waiting for index {{index}} to appear", index=index, cause=e)
            Till(seconds=1).wait()
        Log.alert("Made new index {{index|quote}}", index=index)

        es = Index(kwargs=kwargs, cluster=self)
        return es

    def delete_index(self, index_name):
        if not isinstance(index_name, text_type):
            Log.error("expecting an index name")

        if self.debug:
            Log.note("Deleting index {{index}}", index=index_name)

        # REMOVE ALL ALIASES TOO
        aliases = [a for a in self.get_aliases() if a.index == index_name and a.alias != None]
        if aliases:
            self.post(
                path="/_aliases",
                data={"actions": [{"remove": a} for a in aliases]}
            )

        url = self.settings.host + ":" + text_type(self.settings.port) + "/" + index_name
        try:
            response = http.delete(url)
            if response.status_code != 200:
                Log.error("Expecting a 200, got {{code}}", code=response.status_code)
            details = mo_json.json2value(utf82unicode(response.content))
            if self.debug:
                Log.note("delete response {{response}}", response=details)
            return response
        except Exception as e:
            Log.error("Problem with call to {{url}}", url=url, cause=e)

    def get_aliases(self):
        """
        RETURN LIST OF {"alias":a, "index":i} PAIRS
        ALL INDEXES INCLUDED, EVEN IF NO ALIAS {"alias":Null}
        """
        data = self.get("/_aliases", retry={"times": 5}, timeout=3)
        output = []
        for index, desc in data.items():
            if not desc["aliases"]:
                output.append({"index": index, "alias": None})
            else:
                for a in desc["aliases"]:
                    output.append({"index": index, "alias": a})
        return wrap(output)

    def get_metadata(self, force=False):
        if not self.settings.explore_metadata:
            Log.error("Metadata exploration has been disabled")

        if not self._metadata or force:
            response = self.get("/_cluster/state", retry={"times": 3}, timeout=30)
            with self.metadata_locker:
                self._metadata = wrap(response.metadata)
                # REPLICATE MAPPING OVER ALL ALIASES
                indices = self._metadata.indices
                for i, m in jx.sort(indices.items(), {"value": {"offset": 0}, "sort": -1}):
                    m.index = i
                    for a in m.aliases:
                        if not indices[a]:
                            indices[a] = m
                self.cluster_state = wrap(self.get("/"))
                self.version = self.cluster_state.version.number
            return self._metadata

        return self._metadata

    def post(self, path, **kwargs):
        url = self.settings.host + ":" + text_type(self.settings.port) + path

        try:
            heads = wrap(kwargs).headers
            heads["Accept-Encoding"] = "gzip,deflate"
            heads["Content-Type"] = "application/json"

            data = kwargs.get(b'data')
            if data == None:
                pass
            elif isinstance(data, Mapping):
                kwargs[b'data'] = data = convert.unicode2utf8(value2json(data))
            elif not isinstance(kwargs[b"data"], str):
                Log.error("data must be utf8 encoded string")

            if self.debug:
                sample = kwargs.get(b'data', "")[:300]
                Log.note("{{url}}:\n{{data|indent}}", url=url, data=sample)

            if self.debug:
                Log.note("POST {{url}}", url=url)
            response = http.post(url, **kwargs)
            if response.status_code not in [200, 201]:
                Log.error(response.reason.decode("latin1") + ": " + strings.limit(response.content.decode("latin1"), 100 if self.debug else 10000))
            if self.debug:
                Log.note("response: {{response}}", response=utf82unicode(response.content)[:130])
            details = mo_json.json2value(utf82unicode(response.content))
            if details.error:
                Log.error(convert.quote2string(details.error))
            if details._shards.failed > 0:
                Log.error(
                    "Shard failures {{failures|indent}}",
                    failures=details._shards.failures.reason
                )
            return details
        except Exception as e:
            if url[0:4] != "http":
                suggestion = " (did you forget \"http://\" prefix on the host name?)"
            else:
                suggestion = ""

            if kwargs.get("data"):
                Log.error(
                    "Problem with call to {{url}}" + suggestion + "\n{{body|left(10000)}}",
                    url=url,
                    body=strings.limit(kwargs["data"].decode('utf8'), 100 if self.debug else 10000),
                    cause=e
                )
            else:
                Log.error("Problem with call to {{url}}" + suggestion, url=url, cause=e)

    def delete(self, path, **kwargs):
        url = self.settings.host + ":" + text_type(self.settings.port) + path
        try:
            response = http.delete(url, **kwargs)
            if response.status_code not in [200]:
                Log.error(response.reason+": "+response.all_content)
            if self.debug:
                Log.note("response: {{response}}", response=strings.limit(utf82unicode(response.all_content), 130))
            details = wrap(mo_json.json2value(utf82unicode(response.all_content)))
            if details.error:
                Log.error(details.error)
            return details
        except Exception as e:
            Log.error("Problem with call to {{url}}", url=url, cause=e)

    def get(self, path, **kwargs):
        url = self.settings.host + ":" + text_type(self.settings.port) + path
        try:
            if self.debug:
                Log.note("GET {{url}}", url=url)
            response = http.get(url, **kwargs)
            if response.status_code not in [200]:
                Log.error(response.reason + ": " + response.all_content)
            if self.debug:
                Log.note("response: {{response}}", response=strings.limit(utf82unicode(response.all_content), 130))
            details = wrap(mo_json.json2value(utf82unicode(response.all_content)))
            if details.error:
                Log.error(details.error)
            return details
        except Exception as e:
            Log.error("Problem with call to {{url}}", url=url, cause=e)

    def head(self, path, **kwargs):
        url = self.settings.host + ":" + text_type(self.settings.port) + path
        try:
            response = http.head(url, **kwargs)
            if response.status_code not in [200]:
                Log.error(response.reason+": "+response.all_content)
            if self.debug:
                Log.note("response: {{response}}", response=strings.limit(utf82unicode(response.all_content), 130))
            if response.all_content:
                details = wrap(mo_json.json2value(utf82unicode(response.all_content)))
                if details.error:
                    Log.error(details.error)
                return details
            else:
                return None  # WE DO NOT EXPECT content WITH HEAD REQUEST
        except Exception as e:
            Log.error("Problem with call to {{url}}",  url= url, cause=e)

    def put(self, path, **kwargs):
        url = self.settings.host + ":" + text_type(self.settings.port) + path

        heads = wrap(kwargs).headers
        heads[b"Accept-Encoding"] = b"gzip,deflate"
        heads[b"Content-Type"] = b"application/json"

        data = kwargs.get(b'data')
        if data == None:
            pass
        elif isinstance(data, Mapping):

            kwargs[b'data'] = data = convert.unicode2utf8(convert.value2json(data))
        elif not isinstance(kwargs["data"], str):
            Log.error("data must be utf8 encoded string")

        if self.debug:
            sample = kwargs.get(b'data', "")[:1000]
            Log.note("{{url}}:\n{{data|indent}}", url=url, data=sample)
        try:
            response = http.put(url, **kwargs)
            if response.status_code not in [200]:
                Log.error(response.reason+": "+response.all_content)
            if self.debug:
                Log.note("response: {{response}}",  response= utf82unicode(response.all_content)[0:300:])

            details = mo_json.json2value(utf82unicode(response.content))
            if details.error:
                Log.error(convert.quote2string(details.error))
            if details._shards.failed > 0:
                Log.error("Shard failures {{failures|indent}}",
                          failures="---\n".join(r.replace(";", ";\n") for r in details._shards.failures.reason)
                          )
            return details
        except Exception as e:
            Log.error("Problem with call to {{url}}", url=url, cause=e)


def proto_name(prefix, timestamp=None):
    if not timestamp:
        timestamp = Date.now()
    else:
        timestamp = Date(timestamp)
    return prefix + timestamp.format(INDEX_DATE_FORMAT)


def sort(values):
    return wrap(sorted(values))


def scrub(r):
    """
    REMOVE KEYS OF DEGENERATE VALUES (EMPTY STRINGS, EMPTY LISTS, AND NULLS)
    CONVERT STRINGS OF NUMBERS TO NUMBERS
    RETURNS **COPY**, DOES NOT CHANGE ORIGINAL
    """
    return wrap(_scrub(r))


def _scrub(r):
    try:
        if r == None:
            return None
        elif isinstance(r, (text_type, binary_type)):
            if r == "":
                return None
            return r
        elif Math.is_number(r):
            return convert.value2number(r)
        elif isinstance(r, Mapping):
            if isinstance(r, Data):
                r = object.__getattribute__(r, "_dict")
            output = {}
            for k, v in r.items():
                v = _scrub(v)
                if v != None:
                    output[k.lower()] = v
            if len(output) == 0:
                return None
            return output
        elif hasattr(r, '__iter__'):
            if isinstance(r, FlatList):
                r = r.list
            output = []
            for v in r:
                v = _scrub(v)
                if v != None:
                    output.append(v)
            if not output:
                return None
            elif len(output) == 1:
                return output[0]
            else:
                return output
        else:
            return r
    except Exception as e:
        Log.warning("Can not scrub: {{json}}", json=r, cause=e)


class Alias(Features):
    @override
    def __init__(
        self,
        alias,  # NAME OF THE ALIAS
        type=None,  # SCHEMA NAME, WILL HUNT FOR ONE IF None
        explore_metadata=True,  # IF PROBING THE CLUSTER FOR METADATA IS ALLOWED
        debug=False,
        timeout=None,  # NUMBER OF SECONDS TO WAIT FOR RESPONSE, OR SECONDS TO WAIT FOR DOWNLOAD (PASSED TO requests)
        kwargs=None
    ):
        self.debug = debug
        if self.debug:
            Log.alert("Elasticsearch debugging on {{index|quote}} is on",  index= kwargs.index)
        if alias == None:
            Log.error("Alias can not be None")
        self.settings = kwargs
        self.cluster = Cluster(kwargs)

        if type == None:
            if not explore_metadata:
                Log.error("Alias() was given no `type` (aka schema) and not allowed to explore metadata.  Do not know what to do now.")

            if not self.settings.alias or self.settings.alias==self.settings.index:
                alias_list = self.cluster.get("/_alias")
                candidates = (
                    [(name, i) for name, i in alias_list.items() if self.settings.index in i.aliases.keys()] +
                    [(name, Null) for name, i in alias_list.items() if self.settings.index==name]
                )
                full_name = jx.sort(candidates, 0).last()[0]
                if not full_name:
                    Log.error("No index by name of {{name}}", name=self.settings.index)
                mappings = self.cluster.get("/" + full_name + "/_mapping")[full_name]
            else:
                mappings = self.cluster.get("/"+self.settings.index+"/_mapping")[self.settings.index]

            # FIND MAPPING WITH MOST PROPERTIES (AND ASSUME THAT IS THE CANONICAL TYPE)
            max_prop = -1
            for _type, mapping in mappings.mappings.items():
                if _type == "_default_":
                    continue
                num_prop = len(mapping.properties.keys())
                if max_prop < num_prop:
                    max_prop = num_prop
                    self.settings.type = _type
                    type = _type

            if type == None:
                Log.error("Can not find schema type for index {{index}}", index=coalesce(self.settings.alias, self.settings.index))

        self.path = "/" + alias + "/" + type

    @property
    def url(self):
        return self.cluster.path.rstrip("/") + "/" + self.path.lstrip("/")

    def get_schema(self, retry=True):
        if self.settings.explore_metadata:
            indices = self.cluster.get_metadata().indices
            if not self.settings.alias or self.settings.alias==self.settings.index:
                #PARTIALLY DEFINED settings
                candidates = [(name, i) for name, i in indices.items() if self.settings.index in i.aliases]
                # TODO: MERGE THE mappings OF ALL candidates, DO NOT JUST PICK THE LAST ONE

                index = "dummy value"
                schema = wrap({"_routing": {}, "properties": {}})
                for _, ind in jx.sort(candidates, {"value": 0, "sort": -1}):
                    mapping = ind.mappings[self.settings.type]
                    set_default(schema._routing, mapping._routing)
                    schema.properties = _merge_mapping(schema.properties, mapping.properties)
            else:
                #FULLY DEFINED settings
                index = indices[self.settings.index]
                schema = index.mappings[self.settings.type]

            if index == None and retry:
                #TRY AGAIN, JUST IN CASE
                self.cluster.cluster_state = None
                return self.get_schema(retry=False)

            #TODO: REMOVE THIS BUG CORRECTION
            if not schema and self.settings.type == "test_result":
                schema = index.mappings["test_results"]
            # DONE BUG CORRECTION

            if not schema:
                Log.error(
                    "ElasticSearch index ({{index}}) does not have type ({{type}})",
                    index=self.settings.index,
                    type=self.settings.type
                )
            return schema
        else:
            mapping = self.cluster.get(self.path + "/_mapping")
            if not mapping[self.settings.type]:
                Log.error("{{index}} does not have type {{type}}", self.settings)
            return wrap({"mappings": mapping[self.settings.type]})

    def delete(self, filter):
        self.cluster.get_metadata()

        if self.cluster.cluster_state.version.number.startswith("0.90"):
            query = {"filtered": {
                "query": {"match_all": {}},
                "filter": filter
            }}
        elif self.cluster.cluster_state.version.number.startswith("1."):
            query = {"query": {"filtered": {
                "query": {"match_all": {}},
                "filter": filter
            }}}
        else:
            raise NotImplementedError

        if self.debug:
            Log.note("Delete documents:\n{{query}}", query=query)

        keep_trying = True
        while keep_trying:
            result = self.cluster.delete(
                self.path + "/_query",
                data=value2json(query),
                timeout=60
            )
            keep_trying = False
            for name, status in result._indices.items():
                if status._shards.failed > 0:
                    if status._shards.failures[0].reason.find("rejected execution (queue capacity ") >= 0:
                        keep_trying = True
                        Till(seconds=5).wait()
                        break

            if not keep_trying:
                for name, status in result._indices.items():
                    if status._shards.failed > 0:
                        Log.error(
                            "ES shard(s) report Failure to delete from {{index}}: {{message}}.  Query was {{query}}",
                            index=name,
                            query=query,
                            message=status._shards.failures[0].reason
                        )

    def search(self, query, timeout=None):
        query = wrap(query)
        try:
            if self.debug:
                if len(query.facets.keys()) > 20:
                    show_query = query.copy()
                    show_query.facets = {k: "..." for k in query.facets.keys()}
                else:
                    show_query = query
                Log.note("Query {{path}}\n{{query|indent}}", path=self.path + "/_search", query=show_query)
            return self.cluster.post(
                self.path + "/_search",
                data=query,
                timeout=coalesce(timeout, self.settings.timeout)
            )
        except Exception as e:
            Log.error(
                "Problem with search (path={{path}}):\n{{query|indent}}",
                path=self.path + "/_search",
                query=query,
                cause=e
            )

def parse_properties(parent_index_name, parent_name, esProperties):
    """
    RETURN THE COLUMN DEFINITIONS IN THE GIVEN esProperties OBJECT
    """
    columns = FlatList()
    for name, property in esProperties.items():
        index_name = parent_index_name
        column_name = concat_field(parent_name, name)
        jx_name = column_name
        if split_field(column_name)[-1] == EXISTS_TYPE:
            property.type = "exists"

        if property.type == "nested" and property.properties:
            # NESTED TYPE IS A NEW TYPE DEFINITION
            # MARKUP CHILD COLUMNS WITH THE EXTRA DEPTH
            self_columns = parse_properties(index_name, column_name, property.properties)
            for c in self_columns:
                c.nested_path = [column_name] + c.nested_path
            columns.extend(self_columns)
            columns.append(Column(
                es_index=index_name,
                es_column=column_name,
                names={".": jx_name},
                type="nested",
                nested_path=ROOT_PATH
            ))

            continue

        if property.properties:
            child_columns = parse_properties(index_name, column_name, property.properties)
            columns.extend(child_columns)
            columns.append(Column(
                names={".": jx_name},
                es_index=index_name,
                es_column=column_name,
                nested_path=ROOT_PATH,
                type="source" if property.enabled == False else "object"
            ))

        if property.dynamic:
            continue
        if not property.type:
            continue
        if property.fields:
            child_columns = parse_properties(index_name, column_name, property.fields)
            columns.extend(child_columns)

        if property.type in es_type_to_json_type.keys():
            columns.append(Column(
                es_index=index_name,
                es_column=column_name,
                names={".": jx_name},
                nested_path=ROOT_PATH,
                type=property.type
            ))
            if property.index_name and name != property.index_name:
                columns.append(Column(
                    es_index=index_name,
                    es_column=column_name,
                    names={".": jx_name},
                    nested_path=ROOT_PATH,
                    type=property.type
                ))
        elif property.enabled == None or property.enabled == False:
            columns.append(Column(
                es_index=index_name,
                es_column=column_name,
                names={".": jx_name},
                nested_path=ROOT_PATH,
                type="source" if property.enabled == False else "object"
            ))
        else:
            Log.warning("unknown type {{type}} for property {{path}}", type=property.type, path=query_path)

    return columns


def get_encoder(id_expression="_id"):
    get_id = jx_expression_to_function(id_expression)

    def _encoder(r):
        id = r.get("id")
        r_value = r.get('value')
        if isinstance(r_value, Mapping):
            r_id = get_id(r_value)
            r_value.pop('_id', None)
            if id == None:
                id = r_id
            elif id != r_id and r_id != None:
                Log.error("Expecting id ({{id}}) and _id ({{_id}}) in the record to match", id=id, _id=r._id)
        if id == None:
            id = random_id()

        if "json" in r:
            Log.error("can not handle pure json inserts anymore")
            json = r["json"]
        elif r_value or isinstance(r_value, (dict, Data)):
            json = convert.value2json(r_value)
        else:
            raise Log.error("Expecting every record given to have \"value\" or \"json\" property")

        return {"id": id, "json": json}

    return _encoder




def random_id():
    return Random.hex(40)


def _merge_mapping(a, b):
    """
    MERGE TWO MAPPINGS, a TAKES PRECEDENCE
    """
    for name, b_details in b.items():
        a_details = a[literal_field(name)]
        if a_details.properties and not a_details.type:
            a_details.type = "object"
        if b_details.properties and not b_details.type:
            b_details.type = "object"

        if a_details:
            a_details.type = _merge_type[a_details.type][b_details.type]

            if b_details.type in ES_STRUCT:
                _merge_mapping(a_details.properties, b_details.properties)
        else:
            a[literal_field(name)] = deepcopy(b_details)

    return a


def retro_schema(schema):
    """
    CONVERT SCHEMA FROM 5.x to 1.x
    :param schema:
    :return:
    """
    output = wrap({
        "mappings":{
            typename: {
                "dynamic_templates": (
                    [
                        retro_dynamic_template(*(t.items()[0])) for t in details.dynamic_templates
                    ] + [
                        {
                            "default_strings": {
                                "mapping": {
                                    "index": "not_analyzed",
                                    "type": "keyword",
                                    "store": True
                                },
                                "match_mapping_type": "string",
                                "match": "*"
                            }
                        }
                    ]
                ),
                "properties": retro_properties(details.properties)
            }
            for typename, details in schema.mappings.items()
        },
        "settings": schema.settings
    })
    return output


def retro_dynamic_template(name, template):
    template.mapping.doc_values = True
    if template.mapping.type == "keyword":
        template.mapping.type = "string"
        template.mapping.index = "not_analyzed"
    elif template.mapping.type == "text":
        template.mapping.type = "string"
        template.mapping.index = "analyzed"
    return {name: template}


def retro_properties(properties):
    output = {}
    for k, v in properties.items():
        v.doc_values = True
        v.fielddata = None
        if v.type == "keyword":
            v.type = "string"
            v.index = "not_analyzed"
        elif v.type == "text":
            v.type = "string"
            v.index = "analyzed"
            v.doc_values = None

        if v.properties:
            v.properties = retro_properties(v.properties)

        output[k] = v
    return output


DEFAULT_DYNAMIC_TEMPLATES = wrap([
    {
        "default_boolean": {
            "mapping": {"type": "boolean", "store": True},
            "match": BOOLEAN_TYPE
        }
    },
    {
        "default_number": {
            "mapping": {"type": "double", "store": True},
            "match": NUMBER_TYPE
        }
    },
    {
        "default_string": {
            "mapping": {"type": "keyword", "store": True},
            "match": STRING_TYPE
        }
    },
    {
        "default_exist": {
            "mapping": {"type": "long", "store": True},
            "match": EXISTS_TYPE
        }
    },
    {
        "default_nested": {
            "mapping": {"type": "nested", "store": True},
            "match": NESTED_TYPE
        }
    }
])


es_type_to_json_type = {
    "text": "string",
    "string": "string",
    "keyword": "string",
    "float": "number",
    "double": "number",
    "integer": "number",
    "object": "object",
    "nested": "nested",
    "source": "json",
    "boolean": "boolean",
    "exists": "exists"
}


_merge_type = {
    "boolean": {
        "boolean": "boolean",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "integer": {
        "boolean": "integer",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "long": {
        "boolean": "long",
        "integer": "long",
        "long": "long",
        "float": "double",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "float": {
        "boolean": "float",
        "integer": "float",
        "long": "double",
        "float": "float",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "double": {
        "boolean": "double",
        "integer": "double",
        "long": "double",
        "float": "double",
        "double": "double",
        "string": "string",
        "object": None,
        "nested": None
    },
    "string": {
        "boolean": "string",
        "integer": "string",
        "long": "string",
        "float": "string",
        "double": "string",
        "string": "string",
        "object": None,
        "nested": None
    },
    "object": {
        "boolean": None,
        "integer": None,
        "long": None,
        "float": None,
        "double": None,
        "string": None,
        "object": "object",
        "nested": "nested"
    },
    "nested": {
        "boolean": None,
        "integer": None,
        "long": None,
        "float": None,
        "double": None,
        "string": None,
        "object": "nested",
        "nested": "nested"
    }
}


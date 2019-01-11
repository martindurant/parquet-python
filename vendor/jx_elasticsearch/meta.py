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
from datetime import date, datetime
from decimal import Decimal
import itertools

import jx_base
from jx_base import TableDesc
from jx_base.namespace import Namespace
from jx_base.query import QueryOp
from jx_python import jx
from jx_python.containers.list_usingPythonList import ListContainer
from jx_python.meta import Column, ColumnList
from mo_dots import Data, FlatList, Null, NullType, ROOT_PATH, coalesce, concat_field, is_list, literal_field, relative_field, set_default, split_field, startswith_field, tail_field, wrap
from mo_files import URL
from mo_future import PY2, none_type, text_type
from mo_json import BOOLEAN, EXISTS, INTEGER, OBJECT, STRING, STRUCT
from mo_json.typed_encoder import BOOLEAN_TYPE, EXISTS_TYPE, NUMBER_TYPE, STRING_TYPE, unnest_path, untype_path
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import Except
from mo_logs.strings import quote
from mo_threads import Queue, THREAD_STOP, Thread, Till
from mo_times import Date, HOUR, MINUTE, Timer, WEEK
from pyLibrary.env import elasticsearch
from pyLibrary.env.elasticsearch import _get_best_type_from_mapping, es_type_to_json_type

MAX_COLUMN_METADATA_AGE = 12 * HOUR
ENABLE_META_SCAN = True
DEBUG = False
TOO_OLD = 2*HOUR
OLD_METADATA = MINUTE
TEST_TABLE_PREFIX = "testing"  # USED TO TURN OFF COMPLAINING ABOUT TEST INDEXES


known_clusters = {}  # MAP FROM id(Cluster) TO ElasticsearchMetadata INSTANCE


class ElasticsearchMetadata(Namespace):
    """
    MANAGE SNOWFLAKE SCHEMAS FOR EACH OF THE ALIASES FOUND IN THE CLUSTER
    """

    @override
    def __new__(cls, kwargs, *args, **_kwargs):
        es_cluster = elasticsearch.Cluster(kwargs)
        output = known_clusters.get(id(es_cluster))
        if output is None:
            output = object.__new__(cls)
            known_clusters[id(es_cluster)] = output
        return output

    @override
    def __init__(self, host, index, sql_file='metadata.sqlite', alias=None, name=None, port=9200, kwargs=None):
        if hasattr(self, "settings"):
            return

        self.too_old = TOO_OLD
        self.settings = kwargs
        self.default_name = coalesce(name, alias, index)
        self.es_cluster = elasticsearch.Cluster(kwargs=kwargs)

        self.index_does_not_exist = set()
        self.todo = Queue("refresh metadata", max=100000, unique=True)

        self.index_to_alias = {}

        self.es_metadata = Null
        self.metadata_last_updated = Date.now() - OLD_METADATA

        self.meta = Data()
        self.meta.columns = ColumnList(URL(self.es_cluster.settings.host).host)

        self.alias_to_query_paths = {
            "meta.columns": [ROOT_PATH],
            "meta.tables": [ROOT_PATH]
        }
        self.alias_last_updated = {
            "meta.columns": Date.now(),
            "meta.tables": Date.now()
        }
        table_columns = metadata_tables()
        self.meta.tables = ListContainer(
            "meta.tables",
            [],
            jx_base.Schema(".", table_columns)
        )
        self.meta.columns.extend(table_columns)
        # TODO: fix monitor so it does not bring down ES
        if ENABLE_META_SCAN:
            self.worker = Thread.run("refresh metadata", self.monitor)
        else:
            self.worker = Thread.run("not refresh metadata", self.not_monitor)
        return




    @property
    def namespace(self):
        return self.meta.columns.namespace

    @property
    def url(self):
        return self.es_cluster.url / self.default_name.replace(".", "/")

    def _reload_columns(self, table_desc):
        """
        :param alias: A REAL ALIAS (OR NAME OF INDEX THAT HAS NO ALIAS)
        :return:
        """
        # FIND ALL INDEXES OF ALIAS
        es_last_updated = self.es_cluster.metatdata_last_updated

        alias = table_desc.name
        canonical_index = self.es_cluster.get_best_matching_index(alias).index
        es_metadata_update_required = not (table_desc.timestamp < es_last_updated)
        metadata = self.es_cluster.get_metadata(force=es_metadata_update_required)

        props = [
            (self.es_cluster.get_index(index=i, type=t, debug=DEBUG), t, m.properties)
            for i, d in metadata.indices.items()
            if alias in d.aliases
            for t, m in [_get_best_type_from_mapping(d.mappings)]
        ]

        # CONFIRM ALL COLUMNS ARE SAME, FIX IF NOT
        dirty = False
        all_comparisions = list(jx.pairwise(props)) + list(jx.pairwise(jx.reverse(props)))
        # NOTICE THE SAME (index, type, properties) TRIPLE FROM ABOVE
        for (i1, t1, p1), (i2, t2, p2) in all_comparisions:
            diff = elasticsearch.diff_schema(p2, p1)
            if not self.settings.read_only:
                for d in diff:
                    dirty = True
                    i1.add_property(*d)
        meta = self.es_cluster.get_metadata(force=dirty).indices[canonical_index]

        data_type, mapping = _get_best_type_from_mapping(meta.mappings)
        mapping.properties["_id"] = {"type": "string", "index": "not_analyzed"}
        columns = self._parse_properties(alias, mapping)
        table_desc.timestamp = es_last_updated
        return columns

    def _parse_properties(self, alias, mapping):
        abs_columns = elasticsearch.parse_properties(alias, ".", ROOT_PATH, mapping.properties)
        if DEBUG and any(c.cardinality == 0 and c.name != '_id' for c in abs_columns):
            Log.warning(
                "Some columns are not stored in {{url}} {{index|quote}} table:\n{{names}}",
                url=self.es_cluster.url,
                index=alias,
                names=[
                    ".".join((c.es_index, c.name))
                    for c in abs_columns
                    if c.cardinality == 0
                ]
            )

        with Timer("upserting {{num}} columns", {"num": len(abs_columns)}, silent=not DEBUG):
            # LIST OF EVERY NESTED PATH
            query_paths = [[c.es_column] for c in abs_columns if c.es_type == "nested"]
            for a, b in itertools.product(query_paths, query_paths):
                aa = a[0]
                bb = b[0]
                if aa and bb.startswith(aa):
                    for i, b_prefix in enumerate(b):
                        if len(b_prefix) > len(aa):
                            continue
                        if aa == b_prefix:
                            break  # SPLIT ALREADY FOUND
                        b.insert(i, aa)
                        break
            for q in query_paths:
                q.append(".")
            query_paths.append(ROOT_PATH)

            # ENSURE ALL TABLES HAVE THE QUERY PATHS SET
            self.alias_to_query_paths[alias] = query_paths
            for i, a in self.index_to_alias.items():
                if a == alias:
                    self.alias_to_query_paths[i] = query_paths

            # ENSURE COLUMN HAS CORRECT jx_type
            # PICK DEEPEST NESTED PROPERTY AS REPRESENTATIVE
            output = []
            best = {}
            for abs_column in abs_columns:
                abs_column.jx_type = jx_type(abs_column)
                if abs_column.jx_type not in STRUCT:
                    clean_name = unnest_path(abs_column.name)
                    other = best.get(clean_name)
                    if other:
                        if len(other.nested_path) < len(abs_column.nested_path):
                            output.remove(other)
                            self.meta.columns.update({"clear": ".", "where": {"eq": {"es_column": other.es_column, "es_index": other.es_index}}})
                        else:
                            continue
                    best[clean_name] = abs_column
                output.append(abs_column)

            # REGISTER ALL COLUMNS
            canonicals = []
            for abs_column in output:
                canonical = self.meta.columns.add(abs_column)
                canonicals.append(canonical)

            self.todo.extend(canonicals)
            return canonicals

    def query(self, _query):
        return self.meta.columns.query(QueryOp(set_default(
            {
                "from": self.meta.columns,
                "sort": ["table", "name"]
            },
            _query.__data__()
        )))

    def _find_alias(self, name):
        if self.metadata_last_updated < self.es_cluster.metatdata_last_updated:
            for a in self.es_cluster.get_aliases():
                self.index_to_alias[a.index] = coalesce(a.alias, a.index)
                self.alias_last_updated.setdefault(a.alias, Date.MIN)
        if name in self.alias_last_updated:
            return name
        else:
            return self.index_to_alias.get(name)

    def get_columns(self, table_name, column_name=None, after=None, timeout=None):
        """
        RETURN METADATA COLUMNS

        :param table_name: TABLE WE WANT COLUMNS FOR
        :param column_name:  OPTIONAL NAME, IF INTERESTED IN ONLY ONE COLUMN
        :param after: FORCE LOAD, WAITING FOR last_updated TO BE AFTER THIS TIME
        :param timeout: Signal; True when should give up
        :return:
        """
        DEBUG and after and Log.note("getting columns for after {{time}}", time=after)
        table_path = split_field(table_name)
        root_table_name = table_path[0]

        alias = self._find_alias(root_table_name)
        if not alias:
            self.es_cluster.get_metadata(force=True)
            alias = self._find_alias(root_table_name)
            if not alias:
                Log.error("{{table|quote}} does not exist", table=table_name)

        try:
            table = self.get_table(alias)[0]
            # LAST TIME WE GOT INFO FOR THIS TABLE
            if not table:
                table = TableDesc(
                    name=alias,
                    url=None,
                    query_path=["."],
                    timestamp=Date.MIN
                )
                with self.meta.tables.locker:
                    self.meta.tables.add(table)
                columns = self._reload_columns(table)
                DEBUG and Log.note("columns from reload")
            elif after or table.timestamp < self.es_cluster.metatdata_last_updated:
                columns = self._reload_columns(table)
                DEBUG and Log.note("columns from reload")
            else:
                columns = self.meta.columns.find(alias, column_name)
                DEBUG and Log.note("columns from find()")

            DEBUG and Log.note("columns are {{ids}}", ids=[id(c) for c in columns])

            columns = jx.sort(columns, "name")

            if after is None:
                return columns  # DO NOT WAIT FOR COMPLETE COLUMNS

            # WAIT FOR THE COLUMNS TO UPDATE
            while True:
                pending = [c for c in columns if after >= c.last_updated or (c.cardinality == None and c.jx_type not in STRUCT)]
                if not pending:
                    break
                if timeout:
                    Log.error("trying to gets columns timed out")
                if DEBUG:
                    if len(pending) > 10:
                        Log.note("waiting for {{num}} columns to update by {{timestamp}}", num=len(pending), timestamp=after)
                    else:
                        Log.note("waiting for columns to update by {{timestamp}}; {{columns|json}}", timestamp=after, columns=[c.es_index + "." + c.es_column + " id="+text_type(id(c)) for c in pending])
                Till(seconds=1).wait()
            return columns
        except Exception as e:
            Log.error("Failure to get columns for {{table}}", table=table_name, cause=e)

        return []

    def _update_cardinality(self, column):
        """
        QUERY ES TO FIND CARDINALITY AND PARTITIONS FOR A SIMPLE COLUMN
        """
        now = Date.now()
        if column.es_index in self.index_does_not_exist:
            return

        if column.jx_type in STRUCT:
            Log.error("not supported")
        try:
            if column.es_index == "meta.columns":
                partitions = jx.sort([g[column.es_column] for g, _ in jx.groupby(self.meta.columns, column.es_column) if g[column.es_column] != None])
                self.meta.columns.update({
                    "set": {
                        "partitions": partitions,
                        "count": len(self.meta.columns),
                        "cardinality": len(partitions),
                        "multi": 1,
                        "last_updated": now
                    },
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            if column.es_index == "meta.tables":
                partitions = jx.sort([g[column.es_column] for g, _ in jx.groupby(self.meta.tables, column.es_column) if g[column.es_column] != None])
                self.meta.columns.update({
                    "set": {
                        "partitions": partitions,
                        "count": len(self.meta.tables),
                        "cardinality": len(partitions),
                        "multi": 1,
                        "last_updated": now
                    },
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return

            es_index = column.es_index.split(".")[0]

            is_text = [cc for cc in self.meta.columns if cc.es_column == column.es_column and cc.es_type == "text"]
            if is_text:
                # text IS A MULTIVALUE STRING THAT CAN ONLY BE FILTERED
                result = self.es_cluster.post("/" + es_index + "/_search", data={
                    "aggs": {
                        "count": {"filter": {"match_all": {}}}
                    },
                    "size": 0
                })
                count = result.hits.total
                cardinality = max(1001, count)
                multi = 1001
            elif column.es_column == "_id":
                result = self.es_cluster.post("/" + es_index + "/_search", data={
                    "query": {"match_all": {}},
                    "size": 0
                })
                count = cardinality = result.hits.total
                multi = 1
            elif column.es_type == BOOLEAN:
                result = self.es_cluster.post("/" + es_index + "/_search", data={
                    "aggs": {
                        "count": _counting_query(column)
                    },
                    "size": 0
                })
                count = result.hits.total
                cardinality = 2

                DEBUG and Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "partitions": [False, True],
                        "multi": 1,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            else:
                es_query = {
                    "aggs": {
                        "count": _counting_query(column),
                        "_filter": {
                            "aggs": {"multi": {"max": {"script": "doc[" + quote(column.es_column) + "].values.size()"}}},
                            "filter": {"bool": {"should": [
                                {"range": {"etl.timestamp.~n~": {"gte": (Date.today() - WEEK)}}},
                                {"bool": {"must_not": {"exists": {"field": "etl.timestamp.~n~"}}}}
                            ]}}
                        }
                    },
                    "size": 0
                }

                result = self.es_cluster.post("/" + es_index + "/_search", data=es_query)
                agg_results = result.aggregations
                count = result.hits.total
                cardinality = coalesce(agg_results.count.value, agg_results.count._nested.value, agg_results.count.doc_count)
                multi = int(coalesce(agg_results._filter.multi.value, 1))
                if cardinality == None:
                    Log.error("logic error")

            query = Data(size=0)

            if column.es_column == "_id":
                self.meta.columns.update({
                    "set": {
                        "count": cardinality,
                        "cardinality": cardinality,
                        "multi": 1,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            elif cardinality > 1000 or (count >= 30 and cardinality == count) or (count >= 1000 and cardinality / count > 0.99):
                DEBUG and Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "multi": multi,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            elif column.es_type in elasticsearch.ES_NUMERIC_TYPES and cardinality > 30:
                DEBUG and Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "multi": multi,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            elif len(column.nested_path) != 1:
                query.aggs["_"] = {
                    "nested": {"path": column.nested_path[0]},
                    "aggs": {"_nested": {"terms": {"field": column.es_column}}}
                }
            elif cardinality == 0:  # WHEN DOES THIS HAPPEN?
                query.aggs["_"] = {"terms": {"field": column.es_column}}
            else:
                query.aggs["_"] = {"terms": {"field": column.es_column, "size": cardinality}}

            result = self.es_cluster.post("/" + es_index + "/_search", data=query)

            aggs = result.aggregations._
            if aggs._nested:
                parts = jx.sort(aggs._nested.buckets.key)
            else:
                parts = jx.sort(aggs.buckets.key)

            DEBUG and Log.note("update metadata for {{column.es_index}}.{{column.es_column}} (id={{id}}) at {{time}}", id=id(column), column=column, time=now)
            self.meta.columns.update({
                "set": {
                    "count": count,
                    "cardinality": cardinality,
                    "multi": multi,
                    "partitions": parts,
                    "last_updated": now
                },
                "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
            })
        except Exception as e:
            # CAN NOT IMPORT: THE TEST MODULES SETS UP LOGGING
            # from tests.test_jx import TEST_TABLE
            e = Except.wrap(e)
            TEST_TABLE = "testdata"
            is_missing_index = any(w in e for w in ["IndexMissingException", "index_not_found_exception"])
            is_test_table = column.es_index.startswith((TEST_TABLE_PREFIX, TEST_TABLE))
            if is_missing_index:
                # WE EXPECT TEST TABLES TO DISAPPEAR
                Log.warning("Missing index {{col.es_index}}", col=column, cause=e)
                self.meta.columns.update({
                    "clear": ".",
                    "where": {"eq": {"es_index": column.es_index}}
                })
                self.index_does_not_exist.add(column.es_index)
            elif "No field found for" in e:
                self.meta.columns.update({
                    "clear": ".",
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                Log.warning("Could not get column {{col.es_index}}.{{col.es_column}} info", col=column, cause=e)
            else:
                self.meta.columns.update({
                    "set": {
                        "last_updated": now
                    },
                    "clear": [
                        "count",
                        "cardinality",
                        "multi",
                        "partitions",
                    ],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                Log.warning("Could not get {{col.es_index}}.{{col.es_column}} info", col=column, cause=e)

    def monitor(self, please_stop):
        please_stop.on_go(lambda: self.todo.add(THREAD_STOP))
        while not please_stop:
            try:
                if not self.todo:
                    old_columns = [
                        c
                        for c in self.meta.columns
                        if ((c.last_updated < Date.now() - MAX_COLUMN_METADATA_AGE) or c.cardinality == None) and c.jx_type not in STRUCT
                    ]
                    if old_columns:
                        DEBUG and Log.note(
                            "Old columns {{names|json}} last updated {{dates|json}}",
                            names=wrap(old_columns).es_column,
                            dates=[Date(t).format() for t in wrap(old_columns).last_updated]
                        )
                        self.todo.extend(old_columns)
                    else:
                        DEBUG and Log.note("no more metatdata to update")

                column = self.todo.pop(Till(seconds=(10*MINUTE).seconds))
                if column:
                    if column is THREAD_STOP:
                        continue

                    with Timer("update {{table}}.{{column}}", param={"table": column.es_index, "column": column.es_column}, silent=not DEBUG):
                        if column.es_index in self.index_does_not_exist:
                            DEBUG and Log.note("{{column.es_column}} does not exist", column=column)
                            self.meta.columns.update({
                                "clear": ".",
                                "where": {"eq": {"es_index": column.es_index}}
                            })
                            continue
                        if column.jx_type in STRUCT or split_field(column.es_column)[-1] == EXISTS_TYPE:
                            DEBUG and Log.note("{{column.es_column}} is a struct", column=column)
                            column.last_updated = Date.now()
                            continue
                        elif column.last_updated > Date.now() - TOO_OLD and column.cardinality is not None:
                            # DO NOT UPDATE FRESH COLUMN METADATA
                            DEBUG and Log.note("{{column.es_column}} is still fresh ({{ago}} ago)", column=column, ago=(Date.now()-Date(column.last_updated)).seconds)
                            continue
                        try:
                            self._update_cardinality(column)
                            (DEBUG and not column.es_index.startswith(TEST_TABLE_PREFIX)) and Log.note("updated {{column.name}}", column=column)
                        except Exception as e:
                            if '"status":404' in e:
                                self.meta.columns.update({
                                    "clear": ".",
                                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                                })
                            else:
                                Log.warning("problem getting cardinality for {{column.name}}", column=column, cause=e)
            except Exception as e:
                Log.warning("problem in cardinality monitor", cause=e)

    def not_monitor(self, please_stop):
        Log.alert("metadata scan has been disabled")
        please_stop.on_go(lambda: self.todo.add(THREAD_STOP))
        while not please_stop:
            column = self.todo.pop()
            if column == THREAD_STOP:
                break

            if column.jx_type in STRUCT or split_field(column.es_column)[-1] == EXISTS_TYPE:
                DEBUG and Log.note("{{column.es_column}} is a struct", column=column)
                column.last_updated = Date.now()
                continue
            elif column.last_updated > Date.now() - TOO_OLD and column.cardinality is not None:
                # DO NOT UPDATE FRESH COLUMN METADATA
                DEBUG and Log.note("{{column.es_column}} is still fresh ({{ago}} ago)", column=column, ago=(Date.now()-Date(column.last_updated)).seconds)
                continue

            with Timer("Update {{col.es_index}}.{{col.es_column}}", param={"col": column}, silent=not DEBUG, too_long=0.05):
                if untype_path(column.name) in ["build.type", "run.type"]:
                    try:
                        self._update_cardinality(column)
                    except Exception as e:
                        Log.warning("problem getting cardinality for {{column.name}}", column=column, cause=e)
                else:
                    column.last_updated = Date.now()


    def get_table(self, name):
        if name == "meta.columns":
            return self.meta.columns

        with self.meta.tables.locker:
            return wrap([t for t in self.meta.tables.data if t.name == name])

    def get_snowflake(self, fact_table_name):
        return Snowflake(fact_table_name, self)

    def get_schema(self, name):
        if name == "meta.columns":
            return self.meta.columns.schema
        if name == "meta.tables":
            return self.meta.tables
        root, rest = tail_field(name)
        return self.get_snowflake(root).get_schema(rest)


class Snowflake(object):
    """
    REPRESENT ONE ALIAS, AND ITS NESTED ARRAYS
    """

    def __init__(self, name, namespace):
        self.name = name
        self.namespace = namespace

    def get_schema(self, query_path):
        return Schema(query_path, self)

    @property
    def query_paths(self):
        """
        RETURN A LIST OF ALL NESTED COLUMNS
        """
        output = self.namespace.alias_to_query_paths.get(self.name)
        if output:
            return output
        Log.error("Can not find index {{index|quote}}", index=self.name)

    @property
    def sorted_query_paths(self):
        """
        RETURN A LIST OF ALL SCHEMA'S IN DEPTH-FIRST TOPOLOGICAL ORDER
        """
        return list(reversed(sorted(p[0] for p in self.namespace.alias_to_query_paths.get(self.name))))

    @property
    def columns(self):
        """
        RETURN ALL COLUMNS FROM ORIGIN OF FACT TABLE
        """
        return self.namespace.get_columns(literal_field(self.name))


class Schema(jx_base.Schema):
    """
    REPRESENT JUST ONE TABLE IN A SNOWFLAKE
    """

    def __init__(self, query_path, snowflake):
        if not is_list(snowflake.query_paths[0]):
            Log.error("Snowflake query paths should be a list of string tuples (well, technically, a list of lists of strings)")
        self.snowflake = snowflake
        try:
            path = [
                p
                for p in snowflake.query_paths
                if untype_path(p[0]) == query_path
            ]
            if path:
                # WE DO NOT NEED TO LOOK INTO MULTI-VALUED FIELDS AS A TABLE
                self.multi = None
                self.query_path = path[0]
            else:
                # LOOK INTO A SPECIFIC MULTI VALUED COLUMN
                try:
                    self.multi = [
                        c
                        for c in self.snowflake.columns
                        if untype_path(c.name) == query_path and c.multi > 1
                    ][0]
                    self.query_path = [self.multi.name] + self.multi.nested_path
                except Exception as e:
                    # PROBLEM WITH METADATA UPDATE
                    self.multi = None
                    self.query_path = [query_path] + ["."]

                    Log.warning("Problem getting query path {{path|quote}} in snowflake {{sf|quote}}", path=query_path, sf=snowflake.name, cause=e)

            if not is_list(self.query_path) or self.query_path[len(self.query_path) - 1] != ".":
                Log.error("error")

        except Exception as e:
            Log.error("logic error", cause=e)

    def leaves(self, column_name):
        """
        :param column_name:
        :return: ALL COLUMNS THAT START WITH column_name, NOT INCLUDING DEEPER NESTED COLUMNS
        """
        clean_name = unnest_path(column_name)

        if clean_name != column_name:
            clean_name = column_name
            cleaner = lambda x: x
        else:
            cleaner = unnest_path


        columns = self.columns
        # TODO: '.' IMPLIES ALL FIELDS FROM ABSOLUTE PERPECTIVE, ALL OTHERS ARE A RELATIVE PERSPECTIVE
        # TODO: HOW TO REFER TO FIELDS THAT MAY BE SHADOWED BY A RELATIVE NAME?
        for path in reversed(self.query_path) if clean_name == '.' else self.query_path:
            output = [
                c
                for c in columns
                if (
                    (c.name != "_id" or clean_name == "_id") and
                    (
                        (c.jx_type == EXISTS and column_name.endswith("." + EXISTS_TYPE)) or
                        c.jx_type not in OBJECTS or
                        (clean_name == '.' and c.cardinality == 0)
                    ) and
                    startswith_field(cleaner(relative_field(c.name, path)), clean_name)
                )
            ]
            if output:
                return set(output)
        return set()

    def new_leaves(self, column_name):
        """
        :param column_name:
        :return: ALL COLUMNS THAT START WITH column_name, INCLUDING DEEP COLUMNS
        """
        column_name = unnest_path(column_name)
        columns = self.columns
        all_paths = self.snowflake.sorted_query_paths

        output = {}
        for c in columns:
            if c.name == "_id" and column_name != "_id":
                continue
            if c.jx_type in OBJECTS:
                continue
            if c.cardinality == 0:
                continue
            for path in all_paths:
                if not startswith_field(unnest_path(relative_field(c.name, path)), column_name):
                    continue
                existing = output.get(path)
                if not existing:
                    output[path] = [c]
                    continue
                if len(path) > len(c.nested_path[0]):
                    continue
                if any("." + t + "." in c.es_column for t in (STRING_TYPE, NUMBER_TYPE, BOOLEAN_TYPE)):
                    # ELASTICSEARCH field TYPES ARE NOT ALLOWED
                    continue
                # ONLY THE DEEPEST COLUMN WILL BE CHOSEN
                output[path].append(c)
        return set(output.values())

    def both_leaves(self, column_name):
        old = self.old_leaves(column_name)
        new = self.new_leaves(column_name)

        if old != new:
            Log.error(
                "not the same: {{old}}, {{new}}",
                old=[c.name for c in old],
                new=[c.name for c in new]
            )

        return new

    def values(self, column_name, exclude_type=STRUCT):
        """
        RETURN ALL COLUMNS THAT column_name REFERS TO
        """
        column_name = unnest_path(column_name)
        columns = self.columns
        output = []
        for path in self.query_path:
            full_path = untype_path(concat_field(path, column_name))
            for c in columns:
                if c.jx_type in exclude_type:
                    continue
                # if c.cardinality == 0:
                #     continue
                if untype_path(c.name) == full_path:
                    output.append(c)
            if output:
                return output
        return []

    def __getitem__(self, column_name):
        return self.values(column_name)

    @property
    def name(self):
        return concat_field(self.snowflake.name, self.query_path[0])

    @property
    def columns(self):
        return self.snowflake.columns

    def map_to_es(self):
        """
        RETURN A MAP FROM THE NAMESPACE TO THE es_column NAME
        """
        output = {}
        for path in self.query_path:
            set_default(
                output,
                {
                    k: c.es_column
                    for c in self.columns
                    if c.jx_type not in STRUCT
                    for rel_name in [relative_field(c.name, path)]
                    for k in [rel_name, untype_path(rel_name), unnest_path(rel_name)]
                }
            )
        return output


class Table(jx_base.Table):

    def __init__(self, full_name, container):
        jx_base.Table.__init__(self, full_name)
        self.container=container
        self.schema = container.namespace.get_schema(full_name)


def _counting_query(c):
    if c.es_column == "_id":
        return {"filter": {"match_all": {}}}
    elif len(c.nested_path) != 1:
        return {
            "nested": {
                "path": c.nested_path[0]  # FIRST ONE IS LONGEST
            },
            "aggs": {
                "_nested": {"cardinality": {
                    "field": c.es_column,
                    "precision_threshold": 10 if c.es_type in elasticsearch.ES_NUMERIC_TYPES else 100
                }}
            }
        }
    else:
        return {"cardinality": {
            "field": c.es_column
        }}


def metadata_tables():
    return wrap(
        [
            Column(
                name=c,
                es_index="meta.tables",
                es_column=c,
                es_type="string",
                jx_type=STRING,
                last_updated=Date.now(),
                nested_path=ROOT_PATH
            )
            for c in [
                "name",
                "url",
                "query_path"
            ]
        ]+[
            Column(
                name=c,
                es_index="meta.tables",
                es_column=c,
                es_type="integer",
                jx_type=INTEGER,
                last_updated=Date.now(),
                nested_path=ROOT_PATH
            )
            for c in [
                "timestamp"
            ]
        ]
    )


def jx_type(column):
    """
    return the jx_type for given column
    """
    if column.es_column.endswith(EXISTS_TYPE):
        return EXISTS
    return es_type_to_json_type[column.es_type]


python_type_to_es_type = {
    none_type: "undefined",
    NullType: "undefined",
    bool: "boolean",
    str: "string",
    text_type: "string",
    int: "integer",
    float: "double",
    Data: "object",
    dict: "object",
    set: "nested",
    list: "nested",
    FlatList: "nested",
    Date: "double",
    Decimal: "double",
    datetime: "double",
    date: "double"
}

if PY2:
    python_type_to_es_type[long] = "integer"

_merge_es_type = {
    "undefined": {
        "undefined": "undefined",
        "boolean": "boolean",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "number": "number",
        "string": "string",
        "object": "object",
        "nested": "nested"
    },
    "boolean": {
        "undefined": "boolean",
        "boolean": "boolean",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "number": "number",
        "string": "string",
        "object": None,
        "nested": None
    },
    "integer": {
        "undefined": "integer",
        "boolean": "integer",
        "integer": "integer",
        "long": "long",
        "float": "float",
        "double": "double",
        "number": "number",
        "string": "string",
        "object": None,
        "nested": None
    },
    "long": {
        "undefined": "long",
        "boolean": "long",
        "integer": "long",
        "long": "long",
        "float": "double",
        "double": "double",
        "number": "number",
        "string": "string",
        "object": None,
        "nested": None
    },
    "float": {
        "undefined": "float",
        "boolean": "float",
        "integer": "float",
        "long": "double",
        "float": "float",
        "double": "double",
        "number": "number",
        "string": "string",
        "object": None,
        "nested": None
    },
    "double": {
        "undefined": "double",
        "boolean": "double",
        "integer": "double",
        "long": "double",
        "float": "double",
        "double": "double",
        "number": "number",
        "string": "string",
        "object": None,
        "nested": None
    },
    "number": {
        "undefined": "number",
        "boolean": "number",
        "integer": "number",
        "long": "number",
        "float": "number",
        "double": "number",
        "number": "number",
        "string": "string",
        "object": None,
        "nested": None
    },
    "string": {
        "undefined": "string",
        "boolean": "string",
        "integer": "string",
        "long": "string",
        "float": "string",
        "double": "string",
        "number": "string",
        "string": "string",
        "object": None,
        "nested": None
    },
    "object": {
        "undefined": "object",
        "boolean": None,
        "integer": None,
        "long": None,
        "float": None,
        "double": None,
        "number": None,
        "string": None,
        "object": "object",
        "nested": "nested"
    },
    "nested": {
        "undefined": "nested",
        "boolean": None,
        "integer": None,
        "long": None,
        "float": None,
        "double": None,
        "number": None,
        "string": None,
        "object": "nested",
        "nested": "nested"
    }
}




OBJECTS = (OBJECT, EXISTS)

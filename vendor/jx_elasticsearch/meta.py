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

import itertools
from copy import copy
from itertools import product

from jx_base import STRUCT
from jx_base.query import QueryOp
from jx_base.schema import Schema
from jx_python import jx
from jx_python import meta as jx_base_meta
from jx_python.containers.list_usingPythonList import ListContainer
from jx_python.meta import ColumnList, metadata_columns, metadata_tables, Column, Table
from mo_dots import Data, relative_field, concat_field, SELF_PATH
from mo_dots import coalesce, set_default, Null, split_field, join_field
from mo_dots import wrap
from mo_json.typed_encoder import EXISTS_TYPE
from mo_kwargs import override
from mo_logs import Log
from mo_logs.strings import quote
from mo_threads import Queue
from mo_threads import THREAD_STOP
from mo_threads import Thread
from mo_threads import Till
from mo_times.dates import Date
from mo_times.durations import HOUR, MINUTE
from mo_times.timer import Timer
from pyLibrary.env import elasticsearch
from pyLibrary.env.elasticsearch import es_type_to_json_type

MAX_COLUMN_METADATA_AGE = 12 * HOUR
ENABLE_META_SCAN = False
DEBUG = False
TOO_OLD = 2*HOUR
OLD_METADATA = MINUTE
TEST_TABLE_PREFIX = "testing"  # USED TO TURN OFF COMPLAINING ABOUT TEST INDEXES


class FromESMetadata(Schema):
    """
    QUERY THE METADATA
    """

    def __new__(cls, *args, **kwargs):
        if jx_base_meta.singlton:
            return jx_base_meta.singlton
        else:
            jx_base_meta.singlton = object.__new__(cls)
            return jx_base_meta.singlton

    @override
    def __init__(self, host, index, alias=None, name=None, port=9200, kwargs=None):
        if hasattr(self, "settings"):
            return

        self.too_old = TOO_OLD
        self.settings = kwargs
        self.default_name = coalesce(name, alias, index)
        self.default_es = elasticsearch.Cluster(kwargs=kwargs)
        self.index_does_not_exist = set()
        self.todo = Queue("refresh metadata", max=100000, unique=True)

        self.es_metadata = Null
        self.abs_columns = set()
        self.last_es_metadata = Date.now()-OLD_METADATA

        self.meta=Data()
        table_columns = metadata_tables()
        column_columns = metadata_columns()
        self.meta.tables = ListContainer("meta.tables", [], wrap({c.names["."]: c for c in table_columns}))
        self.meta.columns = ColumnList()
        self.meta.columns.insert(column_columns)
        self.meta.columns.insert(table_columns)
        # TODO: fix monitor so it does not bring down ES
        if ENABLE_META_SCAN:
            self.worker = Thread.run("refresh metadata", self.monitor)
        else:
            self.worker = Thread.run("refresh metadata", self.not_monitor)
        return

    @property
    def query_path(self):
        return None

    @property
    def url(self):
        return self.default_es.path + "/" + self.default_name.replace(".", "/")

    def get_table(self, table_name):
        with self.meta.tables.locker:
            return wrap([t for t in self.meta.tables.data if t.name == table_name])

    def _upsert_column(self, c):
        # ASSUMING THE  self.meta.columns.locker IS HAD
        existing_columns = self.meta.columns.find(c.es_index, c.names["."])
        for canonical in existing_columns:
            if canonical.type == c.type and canonical is not c:
                set_default(c.names, canonical.names)
                for key in Column.__slots__:
                    canonical[key] = c[key]
                if DEBUG:
                    Log.note("todo: {{table}}::{{column}}", table=canonical.es_index, column=canonical.es_column)
                self.todo.add(canonical)
                break
        else:
            self.meta.columns.add(c)
            self.todo.add(c)

            if ENABLE_META_SCAN:
                if DEBUG:
                    Log.note("todo: {{table}}::{{column}}", table=c.es_index, column=c.es_column)
                # MARK meta.columns AS DIRTY TOO
                cols = self.meta.columns.find("meta.columns", None)
                for cc in cols:
                    cc.partitions = cc.cardinality = None
                    cc.last_updated = Date.now() - TOO_OLD
                self.todo.extend(cols)

    def _get_columns(self, table=None):
        # TODO: HANDLE MORE THEN ONE ES, MAP TABLE SHORT_NAME TO ES INSTANCE
        table_path = split_field(table)
        es_index = table_path[0]
        meta = self.es_metadata.indices[es_index]
        if not meta or self.last_es_metadata < Date.now() - OLD_METADATA:
            self.es_metadata = self.default_es.get_metadata(force=True)
            meta = self.es_metadata.indices[es_index]

        for data_type, properties in meta.mappings.items():
            if data_type == "_default_":
                continue
            properties.properties["_id"] = {"type": "string", "index": "not_analyzed"}
            self._parse_properties(meta.index, properties, meta)

    def _parse_properties(self, abs_index, properties, meta):
        # IT IS IMPORTANT THAT NESTED PROPERTIES NAME ALL COLUMNS, AND
        # ALL COLUMNS ARE GIVEN NAMES FOR ALL NESTED PROPERTIES
        def add_column(c, query_path):
            c.last_updated = Date.now() - TOO_OLD
            if query_path[0] != ".":
                c.names[query_path[0]] = relative_field(c.names["."], query_path[0])

            with self.meta.columns.locker:
                for alias in meta.aliases:
                    c_ = copy(c)
                    c_.es_index = alias
                    self._upsert_column(c_)
                self._upsert_column(c)

        abs_columns = elasticsearch.parse_properties(abs_index, None, properties.properties)
        self.abs_columns.update(abs_columns)
        with Timer("upserting {{num}} columns", {"num": len(abs_columns)}, debug=DEBUG):
            # LIST OF EVERY NESTED PATH
            query_paths = [[c.es_column] for c in abs_columns if c.type == "nested"]
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
            query_paths.append(SELF_PATH)

            # ADD RELATIVE COLUMNS
            for abs_column in abs_columns:
                abs_column = abs_column.__copy__()
                abs_column.type = es_type_to_json_type[abs_column.type]
                for query_path in query_paths:
                    add_column(abs_column, query_path)
        pass

    def query(self, _query):
        return self.meta.columns.query(QueryOp(set_default(
            {
                "from": self.meta.columns,
                "sort": ["table", "name"]
            },
            _query.__data__()
        )))

    def get_columns(self, table_name, column_name=None, force=False):
        """
        RETURN METADATA COLUMNS
        """
        table_path = split_field(table_name)
        es_index_name = table_path[0]
        query_path = join_field(table_path[1:])
        table = self.get_table(es_index_name)[0]
        abs_column_name = None if column_name == None else concat_field(query_path, column_name)

        try:
            # LAST TIME WE GOT INFO FOR THIS TABLE
            if not table:
                table = Table(
                    name=es_index_name,
                    url=None,
                    query_path=None,
                    timestamp=Date.now()
                )
                with self.meta.tables.locker:
                    self.meta.tables.add(table)
                self._get_columns(table=es_index_name)
            elif force or table.timestamp == None or table.timestamp < Date.now() - MAX_COLUMN_METADATA_AGE:
                table.timestamp = Date.now()
                self._get_columns(table=es_index_name)

            with self.meta.columns.locker:
                columns = self.meta.columns.find(es_index_name, column_name)
            if columns:
                columns = jx.sort(columns, "names.\.")
                # AT LEAST WAIT FOR THE COLUMNS TO UPDATE
                while len(self.todo) and not all(columns.get("last_updated")):
                    if DEBUG:
                        Log.note("waiting for columns to update {{columns|json}}", columns=[c.es_index+"."+c.es_column for c in columns if not c.last_updated])
                    Till(seconds=1).wait()
                return columns
        except Exception as e:
            Log.error("Not expected", cause=e)

        if abs_column_name:
            Log.error("no columns matching {{table}}.{{column}}", table=table_name, column=abs_column_name)
        else:
            self._get_columns(table=table_name)  # TO TEST WHAT HAPPENED
            Log.error("no columns for {{table}}?!", table=table_name)

    def _update_cardinality(self, column):
        """
        QUERY ES TO FIND CARDINALITY AND PARTITIONS FOR A SIMPLE COLUMN
        """
        if column.es_index in self.index_does_not_exist:
            return

        if column.type in STRUCT:
            Log.error("not supported")
        try:
            if column.es_index == "meta.columns":
                with self.meta.columns.locker:
                    partitions = jx.sort([g[column.es_column] for g, _ in jx.groupby(self.meta.columns, column.es_column) if g[column.es_column] != None])
                    self.meta.columns.update({
                        "set": {
                            "partitions": partitions,
                            "count": len(self.meta.columns),
                            "cardinality": len(partitions),
                            "multi": 1,
                            "last_updated": Date.now()
                        },
                        "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                    })
                return
            if column.es_index == "meta.tables":
                with self.meta.columns.locker:
                    partitions = jx.sort([g[column.es_column] for g, _ in jx.groupby(self.meta.tables, column.es_column) if g[column.es_column] != None])
                    self.meta.columns.update({
                        "set": {
                            "partitions": partitions,
                            "count": len(self.meta.tables),
                            "cardinality": len(partitions),
                            "multi": 1,
                            "last_updated": Date.now()
                        },
                        "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                    })
                return

            es_index = column.es_index.split(".")[0]

            is_text = [cc for cc in self.abs_columns if cc.es_column == column.es_column and cc.type == "text"]
            if is_text:
                # text IS A MULTIVALUE STRING THAT CAN ONLY BE FILTERED
                result = self.default_es.post("/" + es_index + "/_search", data={
                    "aggs": {
                        "count": {"filter": {"match_all": {}}}
                    },
                    "size": 0
                })
                count = result.hits.total
                cardinality = 1001
                multi = 1001
            elif column.es_column == "_id":
                result = self.default_es.post("/" + es_index + "/_search", data={
                    "query": {"match_all": {}},
                    "size": 0
                })
                count = cardinality = result.hits.total
                multi = 1
            else:
                result = self.default_es.post("/" + es_index + "/_search", data={
                    "aggs": {
                        "count": _counting_query(column),
                        "multi": {"max": {"script": "doc[" + quote(column.es_column) + "].values.size()"}}
                    },
                    "size": 0
                })
                r = result.aggregations.count
                count = result.hits.total
                cardinality = coalesce(r.value, r._nested.value, r.doc_count)
                multi = coalesce(r.multi.value, 1)
                if cardinality == None:
                   Log.error("logic error")

            query = Data(size=0)

            if column.es_column == "_id":
                with self.meta.columns.locker:
                    self.meta.columns.update({
                        "set": {
                            "count": cardinality,
                            "cardinality": cardinality,
                            "multi": 1,
                            "last_updated": Date.now()
                        },
                        "clear": ["partitions"],
                        "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                    })
                return
            elif cardinality > 1000 or (count >= 30 and cardinality == count) or (count >= 1000 and cardinality / count > 0.99):
                if DEBUG:
                    Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                with self.meta.columns.locker:
                    self.meta.columns.update({
                        "set": {
                            "count": count,
                            "cardinality": cardinality,
                            "multi": multi,
                            "last_updated": Date.now()
                        },
                        "clear": ["partitions"],
                        "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                    })
                return
            elif column.type in elasticsearch.ES_NUMERIC_TYPES and cardinality > 30:
                if DEBUG:
                    Log.note("{{field}} has {{num}} parts", field=column.es_index, num=cardinality)
                with self.meta.columns.locker:
                    self.meta.columns.update({
                        "set": {
                            "count": count,
                            "cardinality": cardinality,
                            "multi": multi,
                            "last_updated": Date.now()
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
            elif cardinality == 0:
                query.aggs["_"] = {"terms": {"field": column.es_column}}
            else:
                query.aggs["_"] = {"terms": {"field": column.es_column, "size": cardinality}}

            result = self.default_es.post("/" + es_index + "/_search", data=query)

            aggs = result.aggregations._
            if aggs._nested:
                parts = jx.sort(aggs._nested.buckets.key)
            else:
                parts = jx.sort(aggs.buckets.key)

            if DEBUG:
                Log.note("{{field}} has {{parts}}", field=column.names["."], parts=parts)
            with self.meta.columns.locker:
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "multi": multi,
                        "partitions": parts,
                        "last_updated": Date.now()
                    },
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
        except Exception as e:
            # CAN NOT IMPORT: THE TEST MODULES SETS UP LOGGING
            # from tests.test_jx import TEST_TABLE
            TEST_TABLE = "testdata"
            is_missing_index = any(w in e for w in ["IndexMissingException", "index_not_found_exception"])
            is_test_table = any(column.es_index.startswith(t) for t in [TEST_TABLE_PREFIX, TEST_TABLE])
            if is_missing_index and is_test_table:
                # WE EXPECT TEST TABLES TO DISAPPEAR
                with self.meta.columns.locker:
                    self.meta.columns.update({
                        "clear": ".",
                        "where": {"eq": {"es_index": column.es_index}}
                    })
                self.index_does_not_exist.add(column.es_index)
            else:
                self.meta.columns.update({
                    "set": {
                        "last_updated": Date.now()
                    },
                    "clear": [
                        "count",
                        "cardinality",
                        "multi",
                        "partitions",
                    ],
                    "where": {"eq": {"names.\\.": ".", "es_index": column.es_index, "es_column": column.es_column}}
                })
                Log.warning("Could not get {{col.es_index}}.{{col.es_column}} info", col=column, cause=e)

    def monitor(self, please_stop):
        please_stop.on_go(lambda: self.todo.add(THREAD_STOP))
        while not please_stop:
            try:
                if not self.todo:
                    with self.meta.columns.locker:
                        old_columns = [
                            c
                            for c in self.meta.columns
                            if (c.last_updated == None or c.last_updated < Date.now()-TOO_OLD) and c.type not in STRUCT
                        ]
                        if old_columns:
                            if DEBUG:
                                Log.note(
                                    "Old columns {{names|json}} last updated {{dates|json}}",
                                    names=wrap(old_columns).es_column,
                                    dates=[Date(t).format() for t in wrap(old_columns).last_updated]
                                )
                            self.todo.extend(old_columns)
                            # TEST CONSISTENCY
                            for c, d in product(list(self.todo.queue), list(self.todo.queue)):
                                if c.es_column == d.es_column and c.es_index == d.es_index and c != d:
                                    Log.error("")
                        else:
                            if DEBUG:
                                Log.note("no more metatdata to update")

                column = self.todo.pop(Till(seconds=(10*MINUTE).seconds))
                if DEBUG:
                    Log.note("update {{table}}.{{column}}", table=column.es_index, column=column.es_column)
                if column:
                    if column.es_index in self.index_does_not_exist:
                        with self.meta.columns.locker:
                            self.meta.columns.update({
                                "clear": ".",
                                "where": {"eq": {"es_index": column.es_index}}
                            })
                        continue
                    if column.type in STRUCT or column.es_column.endswith("." + EXISTS_TYPE):
                        with self.meta.columns.locker:
                            column.last_updated = Date.now()
                        continue
                    elif column.last_updated >= Date.now()-TOO_OLD:
                        continue
                    try:
                        self._update_cardinality(column)
                        if DEBUG and not column.es_index.startswith(TEST_TABLE_PREFIX):
                            Log.note("updated {{column.name}}", column=column)
                    except Exception as e:
                        Log.warning("problem getting cardinality for {{column.name}}", column=column, cause=e)
            except Exception as e:
                Log.warning("problem in cardinality monitor", cause=e)

    def not_monitor(self, please_stop):
        Log.alert("metadata scan has been disabled")
        please_stop.on_go(lambda: self.todo.add(THREAD_STOP))
        while not please_stop:
            c = self.todo.pop()
            if c == THREAD_STOP:
                break

            if not c.last_updated or c.last_updated >= Date.now()-TOO_OLD:
                continue

            with self.meta.columns.locker:
                self.meta.columns.update({
                    "set": {
                        "last_updated": Date.now()
                    },
                    "clear":[
                        "count",
                        "cardinality",
                        "multi",
                        "partitions",
                    ],
                    "where": {"eq": {"es_index": c.es_index, "es_column": c.es_column}}
                })
            if DEBUG:
                Log.note("Could not get {{col.es_index}}.{{col.es_column}} info", col=c)


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
                    "precision_threshold": 10 if c.type in elasticsearch.ES_NUMERIC_TYPES else 100
                }}
            }
        }
    else:
        return {"cardinality": {
            "field": c.es_column
        }}


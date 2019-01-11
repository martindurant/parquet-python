# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import unicode_literals

from mo_future import is_text, is_binary
from activedata_etl import etl2path, key2etl

from jx_python import jx
from jx_python.containers.list_usingPythonList import ListContainer
from mo_dots import Null, coalesce, wrap
from mo_future import items
from mo_json import CAN_NOT_DECODE_JSON, json2value, value2json
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import Except, suppress_exception
from mo_math.randoms import Random
from mo_threads import Lock
from mo_times.dates import Date, unicode2Date, unix2Date
from mo_times.durations import Duration
from mo_times.timer import Timer
from pyLibrary.aws.s3 import KEY_IS_WRONG_FORMAT, strip_extension
from pyLibrary.env import elasticsearch

MAX_RECORD_LENGTH = 400000
DATA_TOO_OLD = "data is too old to be indexed"
DEBUG = False


class RolloverIndex(object):
    """
    MIMIC THE elasticsearch.Index, WITH EXTRA keys() FUNCTION
    AND THREADED QUEUE AND SPLIT DATA BY
    """
    @override
    def __init__(
        self,
        rollover_field,      # the FIELD with a timestamp to use for determining which index to push to
        rollover_interval,   # duration between roll-over to new index
        rollover_max,        # remove old indexes, do not add old records
        schema,              # es schema
        queue_size=10000,    # number of documents to queue in memory
        batch_size=5000,     # number of documents to push at once
        typed=None,          # indicate if we are expected typed json
        kwargs=None          # plus additional ES settings
    ):
        if kwargs.tjson != None:
            Log.error("not expected")
        if typed == None:
            Log.error("not expected")

        schema.settings.index.max_inner_result_window = 100000  # REQUIRED FOR ACTIVEDATA NESTED QUERIES

        self.settings = kwargs
        self.locker = Lock("lock for rollover_index")
        self.rollover_field = jx.get(rollover_field)
        self.rollover_interval = self.settings.rollover_interval = Duration(rollover_interval)
        self.rollover_max = self.settings.rollover_max = Duration(rollover_max)
        self.known_queues = {}  # MAP DATE TO INDEX
        self.cluster = elasticsearch.Cluster(self.settings)

    def __getattr__(self, item):
        return getattr(self.cluster, item)
        # Log.error("Not supported")

    def _get_queue(self, row):
        row = wrap(row)
        if row.json:
            row.value, row.json = json2value(row.json), None
        timestamp = Date(self.rollover_field(row.value))
        if timestamp == None:
            return Null
        elif timestamp < Date.today() - self.rollover_max:
            return DATA_TOO_OLD

        rounded_timestamp = timestamp.floor(self.rollover_interval)
        with self.locker:
            queue = self.known_queues.get(rounded_timestamp.unix)
        if queue == None:
            candidates = jx.run({
                "from": ListContainer(".", self.cluster.get_aliases()),
                "where": {"regex": {"index": self.settings.index + "\d\d\d\d\d\d\d\d_\d\d\d\d\d\d"}},
                "sort": "index"
            })
            best = None
            for c in candidates:
                c = wrap(c)
                c.date = unicode2Date(c.index[-15:], elasticsearch.INDEX_DATE_FORMAT)
                if timestamp > c.date:
                    best = c
            if not best or rounded_timestamp > best.date:
                if rounded_timestamp < wrap(candidates[-1]).date:
                    es = self.cluster.get_or_create_index(read_only=False, alias=best.alias, index=best.index, kwargs=self.settings)
                else:
                    try:
                        es = self.cluster.create_index(create_timestamp=rounded_timestamp, kwargs=self.settings)
                        es.add_alias(self.settings.index)
                    except Exception as e:
                        e = Except.wrap(e)
                        if "IndexAlreadyExistsException" not in e:
                            Log.error("Problem creating index", cause=e)
                        return self._get_queue(row)  # TRY AGAIN
            else:
                es = self.cluster.get_or_create_index(read_only=False, alias=best.alias, index=best.index, kwargs=self.settings)

            with suppress_exception:
                es.set_refresh_interval(seconds=60 * 5, timeout=5)

            self._delete_old_indexes(candidates)
            threaded_queue = es.threaded_queue(max_size=self.settings.queue_size, batch_size=self.settings.batch_size, silent=True)
            with self.locker:
                queue = self.known_queues[rounded_timestamp.unix] = threaded_queue
        return queue

    def _delete_old_indexes(self, candidates):
        for c in candidates:
            timestamp = unicode2Date(c.index[-15:], "%Y%m%d_%H%M%S")
            if timestamp + self.rollover_interval < Date.today() - self.rollover_max:
                # Log.warning("Will delete {{index}}", index=c.index)
                try:
                    self.cluster.delete_index(c.index)
                except Exception as e:
                    Log.warning("could not delete index {{index}}", index=c.index, cause=e)
        for t, q in items(self.known_queues):
            if unix2Date(t) + self.rollover_interval < Date.today() - self.rollover_max:
                with self.locker:
                    del self.known_queues[t]

        pass

    # ADD keys() SO ETL LOOP CAN FIND WHAT'S GETTING REPLACED
    def keys(self, prefix=None):
        path = jx.reverse(etl2path(key2etl(prefix)))

        if self.cluster.version.startswith(("5.", "6.")):
            stored_fields = "stored_fields"
        else:
            stored_fields = "fields"

        result = self.es.search({
            stored_fields: ["_id"],
            "query": {
                "bool": {
                    "query": {"match_all": {}},
                    "filter": {"and": [{"term": {"etl" + (".source" * i) + ".id": v}} for i, v in enumerate(path)]}
                }
            }
        })

        if result.hits.hits:
            return set(result.hits.hits._id)
        else:
            return set()

    def extend(self, documents, queue=None):
        i = 0
        if queue == None:
            for i, doc in enumerate(documents):
                queue = self._get_queue(doc)
                if queue != None:
                    break
            else:
                Log.note("All documents are too old")
                return

        queue.extend(documents[i::])

    def add(self, doc, queue=None):
        if queue == None:
            queue = self._get_queue(doc)
            if queue == None:
                Log.note("Document not added: Too old")
                return

        queue.add(doc)

    def delete(self, filter):
        self.es.delete(filter)

    def copy(self, keys, source, sample_only_filter=None, sample_size=None, done_copy=None):
        """
        :param keys: THE KEYS TO LOAD FROM source
        :param source: THE SOURCE (USUALLY S3 BUCKET)
        :param sample_only_filter: SOME FILTER, IN CASE YOU DO NOT WANT TO SEND EVERYTHING
        :param sample_size: FOR RANDOM SAMPLE OF THE source DATA
        :param done_copy: CALLBACK, ADDED TO queue, TO FINISH THE TRANSACTION
        :return: LIST OF SUB-keys PUSHED INTO ES
        """
        num_keys = 0
        queue = None
        pending = []  # FOR WHEN WE DO NOT HAVE QUEUE YET
        for key in keys:
            timer = Timer("Process {{key}}", param={"key": key}, silent=not DEBUG)
            try:
                with timer:
                    for rownum, line in enumerate(source.read_lines(strip_extension(key))):
                        if not line:
                            continue

                        if rownum > 0 and rownum % 1000 == 0:
                            Log.note("Ingested {{num}} records from {{key}} in bucket {{bucket}}", num=rownum, key=key, bucket=source.name)

                        insert_me, please_stop = fix(key, rownum, line, source, sample_only_filter, sample_size)
                        if insert_me == None:
                            continue
                        value = insert_me['value']

                        if '_id' not in value:
                            Log.warning("expecting an _id in all S3 records. If missing, there can be duplicates")

                        if queue == None:
                            queue = self._get_queue(insert_me)
                            if queue == None:
                                pending.append(insert_me)
                                if len(pending) > 1000:
                                    if done_copy:
                                        done_copy()
                                    Log.error("first 1000 (key={{key}}) records for {{alias}} have no indication what index to put data", key=tuple(keys)[0], alias=self.settings.index)
                                continue
                            elif queue is DATA_TOO_OLD:
                                break
                            if pending:
                                queue.extend(pending)
                                pending = []

                        num_keys += 1
                        queue.add(insert_me)

                        if please_stop:
                            break
            except Exception as e:
                if KEY_IS_WRONG_FORMAT in e:
                    Log.warning("Could not process {{key}} because bad format. Never trying again.", key=key, cause=e)
                    pass
                elif CAN_NOT_DECODE_JSON in e:
                    Log.warning("Could not process {{key}} because of bad JSON. Never trying again.", key=key, cause=e)
                    pass
                else:
                    Log.warning("Could not process {{key}} after {{duration|round(places=2)}}seconds", key=key, duration=timer.duration.seconds, cause=e)
                    done_copy = None

        if done_copy:
            if queue == None:
                done_copy()
            elif queue is DATA_TOO_OLD:
                done_copy()
            else:
                queue.add(done_copy)

        if [p for p in pending if wrap(p).value.task.state not in ('failed', 'exception')]:
            Log.error("Did not find an index for {{alias}} to place the data for key={{key}}", key=tuple(keys)[0], alias=self.settings.index)

        Log.note("{{num}} keys from {{key|json}} added", num=num_keys, key=keys)
        return num_keys


def fix(source_key, rownum, line, source, sample_only_filter, sample_size):
    """
    :param rownum:
    :param line:
    :param source:
    :param sample_only_filter:
    :param sample_size:
    :return:  (row, no_more_data) TUPLE WHERE row IS {"value":<data structure>} OR {"json":<text line>}
    """
    value = json2value(line)

    if rownum == 0:
        if len(line) > MAX_RECORD_LENGTH:
            _shorten(source_key, value, source)
        value = _fix(value)
        if sample_only_filter and Random.int(int(1.0/coalesce(sample_size, 0.01))) != 0 and jx.filter([value], sample_only_filter):
            # INDEX etl.id==0, BUT NO MORE
            if value.etl.id != 0:
                Log.error("Expecting etl.id==0")
            row = {"value": value}
            return row, True
    elif len(line) > MAX_RECORD_LENGTH:
        _shorten(source_key, value, source)
        value = _fix(value)
    elif '"resource_usage":' in line:
        value = _fix(value)

    row = {"value": value}
    return row, False


def _shorten(source_key, value, source):
    if source.name.startswith("active-data-test-result"):
        value.result.subtests = [s for s in value.result.subtests if s.ok is False]
        value.result.missing_subtests = True
        value.repo.changeset.files = None

    shorter_length = len(value2json(value))
    if shorter_length > MAX_RECORD_LENGTH:
        result_size = len(value2json(value.result))
        if source.name == "active-data-test-result":
            if result_size > MAX_RECORD_LENGTH:
                Log.warning("Epic test failure in {{name}} results in big record for {{id}} of length {{length}}", id=value._id, name=source.name, length=shorter_length)
            else:
                pass  # NOT A PROBLEM
        else:
            Log.warning("Monstrous {{name}} record {{id}} of length {{length}}", id=source_key,  name=source.name, length=shorter_length)


def _fix(value):
    try:
        if value.repo._source:
            value.repo = value.repo._source
        if not value.build.revision12:
            value.build.revision12 = value.build.revision[0:12]
        if value.resource_usage:
            value.resource_usage = None

        return value
    except Exception as e:
        Log.error("unexpected problem", cause=e)

# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

from datetime import date, datetime
import sys

from jx_python import jx
from mo_dots import coalesce, is_data, is_sequence, listwrap, wrap
from mo_future import is_binary, is_text, number_types, text_type
from mo_json import datetime2unix, json2value, value2json
from mo_kwargs import override
from mo_logs import Log, strings
from mo_logs.exceptions import Except, suppress_exception
from mo_logs.log_usingNothing import StructuredLogger
from mo_math.randoms import Random
from mo_threads import Queue, THREAD_STOP, Thread, Till
from mo_times import Duration, MINUTE
from mo_times.dates import datetime2unix
from pyLibrary.convert import bytes2base64
from pyLibrary.env.elasticsearch import Cluster

MAX_BAD_COUNT = 5
LOG_STRING_LENGTH = 2000
PAUSE_AFTER_GOOD_INSERT = 1
PAUSE_AFTER_BAD_INSERT = 60


class StructuredLogger_usingElasticSearch(StructuredLogger):
    @override
    def __init__(
        self,
        host,
        index,
        port=9200,
        type="log",
        queue_size=1000,
        batch_size=100,
        kwargs=None,
    ):
        """
        settings ARE FOR THE ELASTICSEARCH INDEX
        """
        kwargs.timeout = Duration(coalesce(kwargs.timeout, "30second")).seconds
        kwargs.retry.times = coalesce(kwargs.retry.times, 3)
        kwargs.retry.sleep = Duration(coalesce(kwargs.retry.sleep, MINUTE)).seconds
        kwargs.host = Random.sample(listwrap(host), 1)[0]

        schema = json2value(value2json(SCHEMA), leaves=True)
        schema.mappings[type].properties["~N~"].type = "nested"
        self.es = Cluster(kwargs).get_or_create_index(
            schema=schema,
            limit_replicas=True,
            typed=True,
            kwargs=kwargs,
        )
        self.batch_size = batch_size
        self.es.add_alias(coalesce(kwargs.alias, kwargs.index))
        self.queue = Queue("debug logs to es", max=queue_size, silent=True)

        self.worker = Thread.run("add debug logs to es", self._insert_loop)

    def write(self, template, params):
        try:
            params.template = strings.limit(params.template, 2000)
            params.format = None
            self.queue.add({"value": _deep_json_to_string(params, 3)}, timeout=3 * 60)
        except Exception as e:
            sys.stdout.write(text_type(Except.wrap(e)))
        return self

    def _insert_loop(self, please_stop=None):
        bad_count = 0
        while not please_stop:
            try:
                messages = wrap(self.queue.pop_all())
                if not messages:
                    Till(seconds=PAUSE_AFTER_GOOD_INSERT).wait()
                    continue

                for g, mm in jx.groupby(messages, size=self.batch_size):
                    scrubbed = []
                    for i, message in enumerate(mm):
                        if message is THREAD_STOP:
                            please_stop.go()
                            continue
                        try:
                            messages = flatten_causal_chain(message.value)
                            scrubbed.append(
                                {
                                    "value": [
                                        _deep_json_to_string(m, depth=3)
                                        for m in messages
                                    ]
                                }
                            )
                        except Exception as e:
                            Log.warning("Problem adding to scrubbed list", cause=e)

                    self.es.extend(scrubbed)
                    bad_count = 0
            except Exception as f:
                Log.warning("Problem inserting logs into ES", cause=f)
                bad_count += 1
                if bad_count > MAX_BAD_COUNT:
                    Log.warning(
                        "Given up trying to write debug logs to ES index {{index}}",
                        index=self.es.settings.index,
                    )
                Till(seconds=PAUSE_AFTER_BAD_INSERT).wait()

        self.es.flush()

        # CONTINUE TO DRAIN THIS QUEUE
        while not please_stop:
            try:
                Till(seconds=PAUSE_AFTER_GOOD_INSERT).wait()
                self.queue.pop_all()
            except Exception as e:
                Log.warning("Should not happen", cause=e)

    def stop(self):
        with suppress_exception:
            self.queue.add(THREAD_STOP)  # BE PATIENT, LET REST OF MESSAGE BE SENT

        with suppress_exception:
            self.queue.close()
        self.worker.join()


def flatten_causal_chain(log_item, output=None):
    output = output or []

    if is_text(log_item):
        output.append({"template": log_item})
        return

    output.append(log_item)
    for c in listwrap(log_item.cause):
        flatten_causal_chain(c, output)
    log_item.cause = None
    return output


def _deep_json_to_string(value, depth):
    """
    :param value: SOME STRUCTURE
    :param depth: THE MAX DEPTH OF PROPERTIES, DEEPER WILL BE STRING-IFIED
    :return: FLATTER STRUCTURE
    """
    if is_data(value):
        if depth == 0:
            return strings.limit(value2json(value), LOG_STRING_LENGTH)

        return {k: _deep_json_to_string(v, depth - 1) for k, v in value.items()}
    elif is_sequence(value):
        return strings.limit(value2json(value), LOG_STRING_LENGTH)
    elif isinstance(value, number_types):
        return value
    elif is_text(value):
        return strings.limit(value, LOG_STRING_LENGTH)
    elif is_binary(value):
        return strings.limit(bytes2base64(value), LOG_STRING_LENGTH)
    elif isinstance(value, (date, datetime)):
        return datetime2unix(value)
    else:
        return strings.limit(value2json(value), LOG_STRING_LENGTH)


SCHEMA = {
    "settings": {"index.number_of_shards": 2, "index.number_of_replicas": 2},
    "mappings": {
        "_default_": {
            "dynamic_templates": [
                {"everything_else": {"match": "*", "mapping": {"index": False}}}
            ]
        },
    },
}

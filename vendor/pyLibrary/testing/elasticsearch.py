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

import mo_json
from mo_files import File
from mo_logs import Log
from mo_dots import Data
from mo_dots import unwrap, wrap
from pyLibrary import convert
from pyLibrary.env.elasticsearch import Index, Cluster
from mo_kwargs import override
from jx_python import jx


def make_test_instance(name, settings):
    if settings.filename:
        File(settings.filename).delete()
    return open_test_instance(name, settings)


def open_test_instance(name, settings):
    if settings.filename:
        Log.note(
            "Using {{filename}} as {{type}}",
            filename=settings.filename,
            type=name
        )
        return FakeES(settings)
    else:
        Log.note(
            "Using ES cluster at {{host}} as {{type}}",
            host=settings.host,
            type=name
        )
        cluster = Cluster(settings)
        try:
            old_index = cluster.get_index(kwargs=settings)
            old_index.delete()
        except Exception as e:
            if "Can not find index" not in e:
                Log.error("unexpected", cause=e)

        es = cluster.create_index(limit_replicas=True, limit_replicas_warning=False, kwargs=settings)
        es.delete_all_but_self()
        es.add_alias(settings.index)
        return es


class FakeES():
    @override
    def __init__(self, filename, host="fake", index="fake", kwargs=None):
        self.settings = kwargs
        self.filename = kwargs.filename
        try:
            self.data = mo_json.json2value(File(self.filename).read())
        except Exception:
            self.data = Data()

    def search(self, query):
        query = wrap(query)
        f = jx.get(query.query.filtered.filter)
        filtered = wrap([{"_id": i, "_source": d} for i, d in self.data.items() if f(d)])
        if query.fields:
            return wrap({"hits": {"total": len(filtered), "hits": [{"_id": d._id, "fields": unwrap(jx.select([unwrap(d._source)], query.fields)[0])} for d in filtered]}})
        else:
            return wrap({"hits": {"total": len(filtered), "hits": filtered}})

    def extend(self, records):
        """
        JUST SO WE MODEL A Queue
        """
        records = {v["id"]: v["value"] for v in records}

        unwrap(self.data).update(records)

        data_as_json = mo_json.value2json(self.data, pretty=True)

        File(self.filename).write(data_as_json)
        Log.note("{{num}} documents added",  num= len(records))

    def add(self, record):
        if isinstance(record, list):
            Log.error("no longer accepting lists, use extend()")
        return self.extend([record])

    def delete_record(self, filter):
        f = convert.esfilter2where(filter)
        self.data = wrap({k: v for k, v in self.data.items() if not f(v)})

    def set_refresh_interval(self, seconds):
        pass


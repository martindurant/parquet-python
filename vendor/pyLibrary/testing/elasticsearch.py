# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
from jx_python import jx
from mo_dots import Data, Null, is_list, unwrap, wrap
from mo_files import File
import mo_json
from mo_kwargs import override
from mo_logs import Log
from pyLibrary.env.elasticsearch import Cluster


@override
def make_test_instance(name, filename=None, kwargs=None):
    if filename != None:
        File(filename).delete()
    return open_test_instance(kwargs)


@override
def open_test_instance(name, filename=None, es=None, kwargs=None):
    if filename != None:
        Log.note(
            "Using {{filename}} as {{type}}",
            filename=filename,
            type=name
        )
        return FakeES(filename=filename)
    else:
        Log.note(
            "Using ES cluster at {{host}} as {{type}}",
            host=es.host,
            type=name
        )
        cluster = Cluster(es)
        try:
            old_index = cluster.get_index(es)
            cluster.delete_index(old_index.settings.index)
        except Exception as e:
            if "Can not find index" not in e:
                Log.error("unexpected", cause=e)

        output = cluster.create_index(limit_replicas=True, limit_replicas_warning=False, kwargs=es)
        output.delete_all_but_self()
        output.add_alias(es.index)
        return output


class FakeES():
    @override
    def __init__(self, filename, host="fake", index="fake", kwargs=None):
        self.settings = kwargs
        self.file = File(filename)
        self.cluster= Null
        try:
            self.data = mo_json.json2value(self.file.read())
        except Exception as e:
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
        records = {
            v["id"]: v["value"] if "value" in v else mo_json.json2value(v['json'])
            for v in records
        }

        unwrap(self.data).update(records)
        self.refresh()
        Log.note("{{num}} documents added", num=len(records))

    def add(self, record):
        if is_list(record):
            Log.error("no longer accepting lists, use extend()")
        return self.extend([record])

    def delete_record(self, filter):
        f = esfilter2where(filter)
        self.data = wrap({k: v for k, v in self.data.items() if not f(v)})

    def refresh(self, *args, **kwargs):
        data_as_json = mo_json.value2json(self.data, pretty=True)
        self.file.write(data_as_json)


    def set_refresh_interval(self, seconds):
        pass


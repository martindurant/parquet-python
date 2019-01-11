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
from jx_base.container import type2container
from mo_files.url import URL
from mo_kwargs import override
from mo_logs import Log
from pyLibrary.env import http

DEBUG = False

known_hosts = {}


@override
def new_instance(
    host,
    index,
    type=None,
    name=None,
    port=9200,
    read_only=True,
    timeout=None,  # NUMBER OF SECONDS TO WAIT FOR RESPONSE, OR SECONDS TO WAIT FOR DOWNLOAD (PASSED TO requests)
    wait_for_active_shards=1,  # ES WRITE CONSISTENCY (https://www.elastic.co/guide/en/elasticsearch/reference/1.7/docs-index_.html#index-consistency)
    typed=None,
    kwargs=None
):
    try:
        known = known_hosts.get((host, port))
        if known:
            return known(kwargs=kwargs)

        url = URL(host)
        url.port = port
        status = http.get_json(url, stream=False)
        version = status.version.number
        if version.startswith("1."):
            from jx_elasticsearch.es14 import ES14
            type2container.setdefault("elasticsearch", ES14)
            known_hosts[(host, port)] = ES14
            output = ES14(kwargs=kwargs)
            return output
        elif version.startswith(("5.", "6.")):
            from jx_elasticsearch.es52 import ES52
            type2container.setdefault("elasticsearch", ES52)
            known_hosts[(host, port)] = ES52
            output = ES52(kwargs=kwargs)
            return output
        else:
            Log.error("No jx interpreter for Elasticsearch {{version}}", version=version)
    except Exception as e:
        Log.error("Can not make an interpreter for Elasticsearch", cause=e)


# SCRUB THE QUERY SO IT IS VALID
# REPORT ERROR IF OUTPUT APEARS TO HAVE HIT GIVEN limit
def post(es, es_query, limit):
    post_result = None
    try:
        if not es_query.sort:
            es_query.sort = None
        post_result = es.search(es_query)

        for facetName, f in post_result.facets.items():
            if f._type == "statistical":
                continue
            if not f.terms:
                continue

            if not DEBUG and not limit and len(f.terms) == limit:
                Log.error("Not all data delivered (" + str(len(f.terms)) + "/" + str(f.total) + ") try smaller range")
    except Exception as e:
        Log.error("Error with FromES", e)

    return post_result


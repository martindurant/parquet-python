# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import unicode_literals

from collections import Mapping

from jx_base import container
from mo_dots import Data
from mo_dots import wrap, set_default, split_field
from mo_future import text_type
from mo_logs import Log

config = Data()   # config.default IS EXPECTED TO BE SET BEFORE CALLS ARE MADE
_ListContainer = None
_meta = None


def _delayed_imports():
    global _ListContainer
    global _meta


    from jx_python import meta as _meta
    from jx_python.containers.list_usingPythonList import ListContainer as _ListContainer

    _ = _ListContainer
    _ = _meta

    try:
        from pyLibrary.queries.jx_usingMySQL import MySQL
    except Exception:
        MySQL = None

    try:
        from jx_elasticsearch.meta import FromESMetadata
    except Exception:
        FromESMetadata = None

    set_default(container.type2container, {
        "mysql": MySQL,
        "memory": None,
        "meta": FromESMetadata
    })


def wrap_from(frum, schema=None):
    """
    :param frum:
    :param schema:
    :return:
    """
    if not _meta:
        _delayed_imports()

    frum = wrap(frum)

    if isinstance(frum, text_type):
        if not container.config.default.settings:
            Log.error("expecting jx_base.container.config.default.settings to contain default elasticsearch connection info")

        type_ = None
        index = frum
        if frum.startswith("meta."):
            if frum == "meta.columns":
                return _meta.singlton.meta.columns.denormalized()
            elif frum == "meta.tables":
                return _meta.singlton.meta.tables
            else:
                Log.error("{{name}} not a recognized table", name=frum)
        else:
            type_ = container.config.default.type
            index = split_field(frum)[0]

        settings = set_default(
            {
                "index": index,
                "name": frum,
                "exists": True,
            },
            container.config.default.settings
        )
        settings.type = None
        return container.type2container[type_](settings)
    elif isinstance(frum, Mapping) and frum.type and container.type2container[frum.type]:
        # TODO: Ensure the frum.name is set, so we capture the deep queries
        if not frum.type:
            Log.error("Expecting from clause to have a 'type' property")
        return container.type2container[frum.type](frum.settings)
    elif isinstance(frum, Mapping) and (frum["from"] or isinstance(frum["from"], (list, set))):
        from jx_base.query import QueryOp
        return QueryOp.wrap(frum, schema=schema)
    elif isinstance(frum, (list, set)):
        return _ListContainer("test_list", frum)
    else:
        return frum



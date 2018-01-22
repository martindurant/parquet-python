# encoding: utf-8
#
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

from collections import Mapping
from copy import copy

from mo_dots import Data
from mo_dots import set_default, split_field, wrap, join_field
from mo_future import generator_types, text_type
from mo_logs import Log

type2container = Data()
config = Data()   # config.default IS EXPECTED TO BE SET BEFORE CALLS ARE MADE
_ListContainer = None
_Cube = None
_run = None
_Query = None


def _delayed_imports():
    global type2container
    global _ListContainer
    global _Cube
    global _run
    global _Query

    from jx_python.containers.list_usingPythonList import ListContainer as _ListContainer
    from jx_python.containers.cube import Cube as _Cube
    from jx_python.jx import run as _run
    from jx_base.query import QueryOp as _Query

    _ = _ListContainer
    _ = _Cube
    _ = _run
    _ = _Query


class Container(object):
    """
    Containers are data storage capable of handing queries on that storage
    """

    __slots__ = ["data", "namespaces"]

    @classmethod
    def new_instance(type, frum, schema=None):
        """
        Factory!
        """
        if not type2container:
            _delayed_imports()

        if isinstance(frum, Container):
            return frum
        elif isinstance(frum, _Cube):
            return frum
        elif isinstance(frum, _Query):
            return _run(frum)
        elif isinstance(frum, (list, set) + generator_types):
            return _ListContainer(frum)
        elif isinstance(frum, text_type):
            # USE DEFAULT STORAGE TO FIND Container
            if not config.default.settings:
                Log.error("expecting jx_base.container.config.default.settings to contain default elasticsearch connection info")

            settings = set_default(
                {
                    "index": join_field(split_field(frum)[:1:]),
                    "name": frum,
                },
                config.default.settings
            )
            settings.type = None  # WE DO NOT WANT TO INFLUENCE THE TYPE BECAUSE NONE IS IN THE frum STRING ANYWAY
            return type2container["elasticsearch"](settings)
        elif isinstance(frum, Mapping):
            frum = wrap(frum)
            if frum.type and type2container[frum.type]:
                return type2container[frum.type](frum.settings)
            elif frum["from"]:
                frum = copy(frum)
                frum["from"] = Container(frum["from"])
                return _Query.wrap(frum)
            else:
                Log.error("Do not know how to handle {{frum|json}}", frum=frum)
        else:
            Log.error("Do not know how to handle {{type}}", type=frum.__class__.__name__)


    def __init__(self, frum, schema=None):
        object.__init__(self)
        if not type2container:
            _delayed_imports()

        self.data = frum
        if isinstance(schema, list):
            Log.error("expecting map from es_column to column object")

    def query(self, query):
        if query.frum != self:
            Log.error("not expected")
        Log.error("Not implemented")

    def filter(self, where):
        return self.where(where)

    def where(self, where):
        _ = where
        Log.error("not implemented")

    def sort(self, sort):
        _ = sort
        Log.error("not implemented")

    def select(self, select):
        _ = select
        Log.error("not implemented")

    def window(self, window):
        Log.error("not implemented")

    def having(self, having):
        _ = having
        Log.error("not implemented")

    def format(self, format):
        _ = format
        Log.error("not implemented")

    def get_columns(self, table_name):
        """
        USE THE frum TO DETERMINE THE COLUMNS
        """
        Log.error("Not implemented")

    @property
    def schema(self):
        Log.error("Not implemented")

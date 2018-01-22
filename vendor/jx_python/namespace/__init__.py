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
from __future__ import division
from __future__ import absolute_import

from collections import Mapping

from mo_dots import set_default, Data
from jx_base.query import QueryOp


class Namespace(object):

    def convert(self, expr):
        raise NotImplementedError()

    def _convert_query(self, query):
        output = QueryOp("from", None)
        output.select = self._convert_clause(query.select)
        output.where = self.convert(query.where)
        output["from"] = self._convert_from(query["from"])
        output.edges = self._convert_clause(query.edges)
        output.having = convert_list(self._convert_having, query.having)
        output.window = convert_list(self._convert_window, query.window)
        output.sort = self._convert_clause(query.sort)
        output.format = query.format

        return output

    def _convert_from(self, frum):
        raise NotImplementedError()

    def _convert_clause(self, clause):
        raise NotImplementedError()

    def _convert_having(self, clause):
        raise NotImplementedError()

    def _convert_window(self, clause):
        raise NotImplementedError()


def convert_list(operator, operand):
    if operand==None:
        return None
    elif isinstance(operand, Mapping):
        return operator(operand)
    else:
        return map(operator, operand)



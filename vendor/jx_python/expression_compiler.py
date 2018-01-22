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

import re

from pyLibrary import convert
from mo_logs import Log
from mo_dots import coalesce, Data, listwrap, wrap_leaves
from mo_times.dates import Date

true = True
false = False
null = None
EMPTY_DICT = {}


def compile_expression(source):
    """
    THIS FUNCTION IS ON ITS OWN FOR MINIMAL GLOBAL NAMESPACE

    :param source:  PYTHON SOURCE CODE
    :return:  PYTHON FUNCTION
    """

    # FORCE MODULES TO BE IN NAMESPACE
    _ = coalesce
    _ = listwrap
    _ = Date
    _ = convert
    _ = Log
    _ = Data
    _ = EMPTY_DICT
    _ = re
    _ = wrap_leaves

    fake_locals = {}
    try:
        exec(
"""
def output(row, rownum=None, rows=None):
    _source = """ + convert.value2quote(source) + """
    try:
        return """ + source + """
    except Exception as e:
        Log.error("Problem with dynamic function {{func|quote}}",  func=_source, cause=e)
""",
            globals(),
            fake_locals
        )
    except Exception as e:
        Log.error("Bad source: {{source}}", source=source, cause=e)
    return fake_locals['output']

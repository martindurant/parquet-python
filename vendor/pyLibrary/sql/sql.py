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

from mo_dots import wrap
from jx_python.jx_usingMySQL import esfilter2sqlwhere


def find_holes(db, table_name, column_name, _range, filter=None):
    """
    FIND HOLES IN A DENSE COLUMN OF INTEGERS
    RETURNS A LIST OF {"min"min, "max":max} OBJECTS
    """
    if not filter:
        filter = {"match_all": {}}

    _range = wrap(_range)
    params = {
        "min": _range.min,
        "max": _range.max - 1,
        "column_name": db.quote_column(column_name),
        "table_name": db.quote_column(table_name),
        "filter": esfilter2sqlwhere(db, filter)
    }

    min_max = db.query("""
        SELECT
            min({{column_name}}) `min`,
            max({{column_name}})+1 `max`
        FROM
            {{table_name}} a
        WHERE
            a.{{column_name}} BETWEEN {{min}} AND {{max}} AND
            {{filter}}
    """, params)[0]

    db.execute("SET @last={{min}}-1", {"min": _range.min})
    ranges = db.query("""
        SELECT
            prev_rev+1 `min`,
            curr_rev `max`
        FROM (
            SELECT
                a.{{column_name}}-@last diff,
                @last prev_rev,
                @last:=a.{{column_name}} curr_rev
            FROM
                {{table_name}} a
            WHERE
                a.{{column_name}} BETWEEN {{min}} AND {{max}} AND
                {{filter}}
            ORDER BY
                a.{{column_name}}
        ) a
        WHERE
            diff>1
    """, params)

    if ranges:
        ranges.append({"min": min_max.max, "max": _range.max})
    else:
        if min_max.min:
            ranges.append({"min": _range.min, "max": min_max.min})
            ranges.append({"min": min_max.max, "max": _range.max})
        else:
            ranges.append(_range)

    return ranges


def values2rows(values, column_names):
    """
     CONVERT LIST OF JSON-IZABLE DATA STRUCTURE TO DATABASE ROW
     value - THE STRUCTURE TO CONVERT INTO row
     column_names - FOR ORDERING THE ALLOWED COLUMNS (EXTRA ATTRIBUTES ARE
                    LOST) THE COLUMN NAMES ARE EXPECTED TO HAVE dots (.)
                    FOR DEEPER PROPERTIES
    """
    values = wrap(values)
    lookup = {name: i for i, name in enumerate(column_names)}
    output = []
    for value in values:
        row = [None] * len(column_names)
        for k, v in value.leaves():
            index = lookup.get(k, -1)
            if index != -1:
                row[index] = v
        output.append(row)
    return output











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

from datetime import datetime

from mo_logs import Log, strings
from mo_dots import Data
from mo_dots import coalesce
from mo_dots import wrap
from mo_dots.lists import FlatList
from pyLibrary import convert
from mo_math import COUNT
from mo_math import Math
from mo_math import stats
from jx_base import domains
from jx_elasticsearch.es09.expressions import value2MVEL, isKeyword
from mo_times import durations

TrueFilter = {"match_all": {}}
DEBUG = False

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


def build_es_query(query):
    output = wrap({
        "query": {"match_all": {}},
        "from": 0,
        "size": 100 if DEBUG else 0,
        "sort": [],
        "facets": {
        }
    })

    if DEBUG:
        # TO LIMIT RECORDS TO WHAT'S IN FACETS
        output.query = {
            "bool": {
                "query": {
                    "match_all": {}
                },
                "filter": query.where.to_esfilter()
            }
        }

    return output





def compileTime2Term(edge):
    """
    RETURN MVEL CODE THAT MAPS TIME AND DURATION DOMAINS DOWN TO AN INTEGER AND
    AND THE JAVASCRIPT THAT WILL TURN THAT INTEGER BACK INTO A PARTITION (INCLUDING NULLS)
    """
    if edge.esscript:
        Log.error("edge script not supported yet")

    # IS THERE A LIMIT ON THE DOMAIN?
    numPartitions = len(edge.domain.partitions)
    value = edge.value
    if isKeyword(value):
        value = "doc[\"" + value + "\"].value"

    nullTest = compileNullTest(edge)
    ref = coalesce(edge.domain.min, edge.domain.max, datetime(2000, 1, 1))

    if edge.domain.interval.month > 0:
        offset = ref.subtract(ref.floorMonth(), durations.DAY).milli
        if offset > durations.DAY.milli * 28:
            offset = ref.subtract(ref.ceilingMonth(), durations.DAY).milli
        partition2int = "milli2Month(" + value + ", " + value2MVEL(offset) + ")"
        partition2int = "((" + nullTest + ") ? 0 : " + partition2int + ")"

        def int2Partition(value):
            if Math.round(value) == 0:
                return edge.domain.NULL

            d = datetime(str(value)[:4:], str(value)[-2:], 1)
            d = d.addMilli(offset)
            return edge.domain.getPartByKey(d)
    else:
        partition2int = "Math.floor((" + value + "-" + value2MVEL(ref) + ")/" + edge.domain.interval.milli + ")"
        partition2int = "((" + nullTest + ") ? " + numPartitions + " : " + partition2int + ")"

        def int2Partition(value):
            if Math.round(value) == numPartitions:
                return edge.domain.NULL
            return edge.domain.getPartByKey(ref.add(edge.domain.interval.multiply(value)))

    return Data(toTerm={"head": "", "body": partition2int}, fromTerm=int2Partition)


# RETURN MVEL CODE THAT MAPS DURATION DOMAINS DOWN TO AN INTEGER AND
# AND THE JAVASCRIPT THAT WILL TURN THAT INTEGER BACK INTO A PARTITION (INCLUDING NULLS)
def compileDuration2Term(edge):
    if edge.esscript:
        Log.error("edge script not supported yet")

    # IS THERE A LIMIT ON THE DOMAIN?
    numPartitions = len(edge.domain.partitions)
    value = edge.value
    if isKeyword(value):
        value = "doc[\"" + value + "\"].value"

    ref = coalesce(edge.domain.min, edge.domain.max, durations.ZERO)
    nullTest = compileNullTest(edge)

    ms = edge.domain.interval.milli
    if edge.domain.interval.month > 0:
        ms = durations.YEAR.milli / 12 * edge.domain.interval.month

    partition2int = "Math.floor((" + value + "-" + value2MVEL(ref) + ")/" + ms + ")"
    partition2int = "((" + nullTest + ") ? " + numPartitions + " : " + partition2int + ")"

    def int2Partition(value):
        if Math.round(value) == numPartitions:
            return edge.domain.NULL
        return edge.domain.getPartByKey(ref.add(edge.domain.interval.multiply(value)))

    return Data(toTerm={"head": "", "body": partition2int}, fromTerm=int2Partition)


# RETURN MVEL CODE THAT MAPS THE numeric DOMAIN DOWN TO AN INTEGER AND
# AND THE JAVASCRIPT THAT WILL TURN THAT INTEGER BACK INTO A PARTITION (INCLUDING NULLS)
def compileNumeric2Term(edge):
    if edge.script:
        Log.error("edge script not supported yet")

    if edge.domain.type != "numeric" and edge.domain.type != "count":
        Log.error("can only translate numeric domains")

    numPartitions = len(edge.domain.partitions)
    value = edge.value
    if isKeyword(value):
        value = "doc[\"" + value + "\"].value"

    if not edge.domain.max:
        if not edge.domain.min:
            ref = 0
            partition2int = "Math.floor(" + value + ")/" + value2MVEL(edge.domain.interval) + ")"
            nullTest = "false"
        else:
            ref = value2MVEL(edge.domain.min)
            partition2int = "Math.floor((" + value + "-" + ref + ")/" + value2MVEL(edge.domain.interval) + ")"
            nullTest = "" + value + "<" + ref
    elif not edge.domain.min:
        ref = value2MVEL(edge.domain.max)
        partition2int = "Math.floor((" + value + "-" + ref + ")/" + value2MVEL(edge.domain.interval) + ")"
        nullTest = "" + value + ">=" + ref
    else:
        top = value2MVEL(edge.domain.max)
        ref = value2MVEL(edge.domain.min)
        partition2int = "Math.floor((" + value + "-" + ref + ")/" + value2MVEL(edge.domain.interval) + ")"
        nullTest = "(" + value + "<" + ref + ") or (" + value + ">=" + top + ")"

    partition2int = "((" + nullTest + ") ? " + numPartitions + " : " + partition2int + ")"
    offset = convert.value2int(ref)

    def int2Partition(value):
        if Math.round(value) == numPartitions:
            return edge.domain.NULL
        return edge.domain.getPartByKey((value * edge.domain.interval) + offset)

    return Data(toTerm={"head": "", "body": partition2int}, fromTerm=int2Partition)


def compileString2Term(edge):
    if edge.esscript:
        Log.error("edge script not supported yet")

    value = edge.value
    if isKeyword(value):
        value = strings.expand_template("getDocValue({{path}})", {"path": quote(value)})
    else:
        Log.error("not handled")

    def fromTerm(value):
        return edge.domain.getPartByKey(value)

    return Data(
        toTerm={"head": "", "body": value},
        fromTerm=fromTerm
    )


def compileNullTest(edge):
    """
    RETURN A MVEL EXPRESSION THAT WILL EVALUATE TO true FOR OUT-OF-BOUNDS
    """
    if edge.domain.type not in domains.ALGEBRAIC:
        Log.error("can only translate time and duration domains")

    # IS THERE A LIMIT ON THE DOMAIN?
    value = edge.value
    if isKeyword(value):
        value = "doc[\"" + value + "\"].value"

    if not edge.domain.max:
        if not edge.domain.min:
            return False
        bot = value2MVEL(edge.domain.min)
        nullTest = "" + value + "<" + bot
    elif not edge.domain.min:
        top = value2MVEL(edge.domain.max)
        nullTest = "" + value + ">=" + top
    else:
        top = value2MVEL(edge.domain.max)
        bot = value2MVEL(edge.domain.min)
        nullTest = "(" + value + "<" + bot + ") or (" + value + ">=" + top + ")"

    return nullTest


def compileEdges2Term(mvel_compiler, edges, constants):
    """
    TERMS ARE ALWAYS ESCAPED SO THEY CAN BE COMPOUNDED WITH PIPE (|)

    GIVE MVEL CODE THAT REDUCES A UNIQUE TUPLE OF PARTITIONS DOWN TO A UNIQUE TERM
    GIVE LAMBDA THAT WILL CONVERT THE TERM BACK INTO THE TUPLE
    RETURNS TUPLE OBJECT WITH "type" and "value" ATTRIBUTES.
    "type" CAN HAVE A VALUE OF "script", "field" OR "count"
    CAN USE THE constants (name, value pairs)
    """

    # IF THE QUERY IS SIMPLE ENOUGH, THEN DO NOT USE TERM PACKING
    edge0 = edges[0]

    if len(edges) == 1 and edge0.domain.type in ["set", "default"]:
        # THE TERM RETURNED WILL BE A MEMBER OF THE GIVEN SET
        def temp(term):
            return FlatList([edge0.domain.getPartByKey(term)])

        if edge0.value and isKeyword(edge0.value):
            return Data(
                field=edge0.value,
                term2parts=temp
            )
        elif COUNT(edge0.domain.dimension.fields) == 1:
            return Data(
                field=edge0.domain.dimension.fields[0],
                term2parts=temp
            )
        elif not edge0.value and edge0.domain.partitions:
            script = mvel_compiler.Parts2TermScript(edge0.domain)
            return Data(
                expression=script,
                term2parts=temp
            )
        else:
            return Data(
                expression=mvel_compiler.compile_expression(edge0.value, constants),
                term2parts=temp
            )

    mvel_terms = []     # FUNCTION TO PACK TERMS
    fromTerm2Part = []  # UNPACK TERMS BACK TO PARTS
    for e in edges:
        domain = e.domain
        fields = domain.dimension.fields

        if not e.value and fields:
            code, decode = mvel_compiler.Parts2Term(e.domain)
            t = Data(
                toTerm=code,
                fromTerm=decode
            )
        elif fields:
            Log.error("not expected")
        elif e.domain.type == "time":
            t = compileTime2Term(e)
        elif e.domain.type == "duration":
            t = compileDuration2Term(e)
        elif e.domain.type in domains.ALGEBRAIC:
            t = compileNumeric2Term(e)
        elif e.domain.type == "set" and not fields:
            def fromTerm(term):
                return e.domain.getPartByKey(term)

            code, decode = mvel_compiler.Parts2Term(e.domain)
            t = Data(
                toTerm=code,
                fromTerm=decode
            )
        else:
            t = compileString2Term(e)

        if not t.toTerm.body:
            mvel_compiler.Parts2Term(e.domain)
            Log.unexpected("what?")

        fromTerm2Part.append(t.fromTerm)
        mvel_terms.append(t.toTerm.body)

    # REGISTER THE DECODE FUNCTION
    def temp(term):
        terms = term.split('|')

        output = FlatList([t2p(t) for t, t2p in zip(terms, fromTerm2Part)])
        return output

    return Data(
        expression=mvel_compiler.compile_expression("+'|'+".join(mvel_terms), constants),
        term2parts=temp
    )


def fix_es_stats(s):
    """
    ES RETURNS BAD DEFAULT VALUES FOR STATS
    """
    s = wrap(s)
    if s.count == 0:
        return stats.zero
    return s


# MAP NAME TO SQL FUNCTION
aggregates = {
    "none": "none",
    "one": "count",
    "sum": "total",
    "add": "total",
    "count": "count",
    "maximum": "max",
    "minimum": "min",
    "max": "max",
    "min": "min",
    "mean": "mean",
    "average": "mean",
    "avg": "mean",
    "N": "count",
    "X0": "count",
    "X1": "total",
    "X2": "sum_of_squares",
    "std": "std_deviation",
    "stddev": "std_deviation",
    "var": "variance",
    "variance": "variance"
}



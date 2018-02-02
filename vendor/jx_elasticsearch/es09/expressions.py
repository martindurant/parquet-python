# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from collections import Mapping

from datetime import datetime
import re

from mo_future import text_type
from pyLibrary import convert
from mo_collections import reverse
from mo_logs import Log
from mo_logs.strings import quote
from mo_math import Math
from mo_dots import split_field, Data, Null, join_field, coalesce, listwrap
from mo_times.durations import Duration


class _MVEL(object):
    def __init__(self, fromData, isLean=False):
        self.fromData = fromData
        self.isLean = isLean
        self.prefixMap = []
        self.functions = {}


    def code(self, query):
        """
        RETURN THE MVEL THAT WILL FILTER USING query.where AND TERM-PACK THE query.select CLAUSE
        """
        selectList = listwrap(query.select)
        fromPath = query.frum.name  # FIRST NAME IS THE INDEX
        sourceVar = "__sourcedoc__"
        whereClause = query.where

        # PARSE THE fromPath
        code = self.frum(fromPath, sourceVar, "__loop")
        select = self.select(selectList, fromPath, "output", sourceVar)

        body = "var output = \"\";\n" + \
               code.replace(
                   "<CODE>",
                   "if (" + _where(whereClause, lambda v: self._translate(v)) + "){\n" +
                   select.body +
                   "}\n"
               ) + \
               "output\n"

        # ADD REFERENCED CONTEXT VARIABLES
        context = self.getFrameVariables(body)

        func = UID()
        predef = addFunctions(select.head+context+body).head
        param = "_source" if body.find(sourceVar) else ""

        output = predef + \
            select.head + \
            context + \
            'var ' + func + ' = function('+sourceVar+'){\n' + \
            body + \
            '};\n' + \
            func + '('+param+')\n'

        return Compiled(output)

    def frum(self, fromPath, sourceVar, loopVariablePrefix):
        """
        indexName NAME USED TO REFER TO HIGH LEVEL DOCUMENT
        loopVariablePrefix PREFIX FOR LOOP VARIABLES
        """
        loopCode = "if (<PATH> != null){ for(<VAR> : <PATH>){\n<CODE>\n}}\n"
        self.prefixMap = []
        code = "<CODE>"
        path = split_field(fromPath)

        # ADD LOCAL VARIABLES
        columns = INDEX_CACHE[path[0]].columns
        for i, c in enumerate(columns):
            if c.name.find("\\.") >= 0:
                self.prefixMap.insert(0, {
                    "path": c.name,
                    "variable": "get(" + sourceVar + ", \"" + c.name.replace("\\.", ".") + "\")"
                })
            else:
                self.prefixMap.insert(0, {
                    "path": c.name,
                    "variable": sourceVar + ".?" + c.name
                })

        # ADD LOOP VARIABLES
        currPath = []
        # self.prefixMap.insert(0, {"path": path[0], "variable": path[0]})
        for i, step in enumerate(path[1::]):
            loopVariable = loopVariablePrefix + str(i)
            currPath.append(step)
            pathi = ".".join(currPath)
            shortPath = self._translate(pathi)
            self.prefixMap.insert(0, {"path": pathi, "variable": loopVariable})

            loop = loopCode.replace("<VAR>", loopVariable).replace("<PATH>", shortPath)
            code = code.replace("<CODE>", loop)
        return code

    def _translate(self, variableName):
        shortForm = variableName
        for p in self.prefixMap:
            prefix = p["path"]
            if shortForm == prefix:
                shortForm = p["variable"]
            else:
                shortForm = replacePrefix(shortForm, prefix + ".", p["variable"] + ".?")  # ADD NULL CHECK
                shortForm = replacePrefix(shortForm, prefix + "[", p["variable"] + "[")
        return shortForm

    #  CREATE A PIPE DELIMITED RESULT SET
    def select(self, selectList, fromPath, varName, sourceVar):
        path = split_field(fromPath)
        is_deep = len(path) > 1
        heads = []
        list = []
        for s in selectList:
            if is_deep:
                if s.value and isKeyword(s.value):
                    shortForm = self._translate(s.value)
                    list.append("Value2Pipe(" + shortForm + ")\n")
                else:
                    Log.error("do not know how to handle yet")
            else:
                if s.value and isKeyword(s.value):
                    list.append("Value2Pipe(getDocValue(" + value2MVEL(s.value) + "))\n")
                elif s.value:
                    shortForm = self._translate(s.value)
                    list.append("Value2Pipe(" + shortForm + ")\n")
                else:
                    code, decode = self.Parts2Term(s.domain)
                    heads.append(code.head)
                    list.append("Value2Pipe(" + code.body + ")\n")


        if len(split_field(fromPath)) > 1:
            output = 'if (' + varName + ' != "") ' + varName + '+="|";\n' + varName + '+=' + '+"|"+'.join(["Value2Pipe("+v+")\n" for v in list]) + ';\n'
        else:
            output = varName + ' = ' + '+"|"+'.join(["Value2Pipe("+v+")\n" for v in list]) + ';\n'

        return Data(
            head="".join(heads),
            body=output
        )
    def Parts2Term(self, domain):
        """
        TERMS ARE ALWAYS ESCAPED SO THEY CAN BE COMPOUNDED WITH PIPE (|)

        CONVERT AN ARRAY OF PARTS{name, esfilter} TO AN MVEL EXPRESSION
        RETURN expression, function PAIR, WHERE
            expression - MVEL EXPRESSION
            function - TAKES RESULT OF expression AND RETURNS PART
        """
        fields = domain.dimension.fields

        term = []
        if len(split_field(self.fromData.name)) == 1 and fields:
            if isinstance(fields, Mapping):
                # CONVERT UNORDERED FIELD DEFS
                jx_fields, es_fields = zip(*[(k, fields[k]) for k in sorted(fields.keys())])
            else:
                jx_fields, es_fields = zip(*[(i, e) for i, e in enumerate(fields)])

            # NO LOOPS BECAUSE QUERY IS SHALLOW
            # DOMAIN IS FROM A DIMENSION, USE IT'S FIELD DEFS TO PULL
            if len(es_fields) == 1:
                def fromTerm(term):
                    return domain.getPartByKey(term)

                return Data(
                    head="",
                    body='getDocValue('+quote(domain.dimension.fields[0])+')'
                ), fromTerm
            else:
                def fromTerm(term):
                    terms = [convert.pipe2value(t) for t in convert.pipe2value(term).split("|")]

                    candidate = dict(zip(jx_fields, terms))
                    for p in domain.partitions:
                        for k, t in candidate.items():
                            if p.value[k] != t:
                                break
                        else:
                            return p
                    if domain.type in ["uid", "default"]:
                        part = {"value": candidate}
                        domain.partitions.append(part)
                        return part
                    else:
                        return Null

                for f in es_fields:
                    term.append('Value2Pipe(getDocValue('+quote(f)+'))')

                return Data(
                    head="",
                    body='Value2Pipe('+('+"|"+'.join(term))+')'
                ), fromTerm
        else:
            for v in domain.partitions:
                term.append("if (" + _where(v.esfilter, lambda x: self._translate(x)) + ") " + value2MVEL(domain.getKey(v)) + "; else ")
            term.append(value2MVEL(domain.getKey(domain.NULL)))

            func_name = "_temp"+UID()
            return self.register_function("+\"|\"+".join(term))

    def Parts2TermScript(self, domain):
        code, decode = self.Parts2Term(domain)
        func = addFunctions(code.head + code.body)
        return func.head + code.head + code.body, decode

    def getFrameVariables(self, body):
        contextVariables = []
        columns = self.fromData.columns

        parentVarNames = set()    # ALL PARENTS OF VARIABLES WITH "." IN NAME
        body = body.replace(".?", ".")

        for i, c in enumerate(columns):
            j = body.find(c.name, 0)
            while j >= 0:
                s = j
                j = body.find(c.name, s + 1)

                test0 = body[s - 1: s + len(c.name) + 1:]
                test3 = body[s - 8: s + len(c.name):]

                if test0[:-1] == "\"" + c.name:
                    continue
                if test3 == "_source." + c.name:
                    continue

                def defParent(name):
                    # DO NOT MAKE THE SAME PARENT TWICE
                    if name in parentVarNames:
                        return
                    parentVarNames.add(name)

                    if len(split_field(name)) == 1:
                        contextVariables.append("Map " + name + " = new HashMap();\n")
                    else:
                        defParent(join_field(split_field(name)[0:-1]))
                        contextVariables.append(name + " = new HashMap();\n")

                body = body.replace(c.name, "-"*len(c.name))

                if self.isLean or c.useSource:
                    if len(split_field(c.name)) > 1:
                        defParent(join_field(split_field(c.name)[0:-1]))
                        contextVariables.append(c.name + " = getSourceValue(\"" + c.name + "\");\n")
                    else:
                        contextVariables.append(c.name + " = _source[\"" + c.name + "\"];\n")
                else:
                    if len(split_field(c.name)) > 1:
                        defParent(join_field(split_field(c.name)[0:-1]))
                        contextVariables.append(c.name + " = getDocValue(\"" + c.name + "\");\n")
                    else:
                        contextVariables.append(c.name + " = getDocValue(\"" + c.name + "\");\n")
                break

        return "".join(contextVariables)

    def compile_expression(self, expression, constants=None):
        # EXPAND EXPRESSION WITH ANY CONSTANTS
        expression = setValues(expression, constants)

        fromPath = self.fromData.name           # FIRST NAME IS THE INDEX
        indexName = join_field(split_field(fromPath)[:1:])

        context = self.getFrameVariables(expression)
        if context == "":
            return addFunctions(expression).head+expression

        func = UID()
        code = addFunctions(context+expression)
        output = code.head + \
            'var ' + func + ' = function(' + indexName + '){\n' + \
            context + \
            expression + ";\n" + \
            '};\n' + \
            func + '(_source)\n'

        return Compiled(output)

    def register_function(self, code):
        for n, c in self.functions.items():
            if c == code:
                break
        else:
            n = "_temp" + UID()
            self.functions[n] = code

        return Data(
            head='var ' + n + ' = function(){\n' + code + '\n};\n',
            body=n + '()\n'
        )


class Compiled(object):
    def __init__(self, code):
        self.code=code

    def __str__(self):
        return self.code

    def __data__(self):
        return self.code




__UID__ = 1000


def UID():
    output = "_" + str(__UID__)
    globals()["__UID__"] += 1
    return output


def setValues(expression, constants):
    if not constants:
        return expression

    constants = constants.copy()

    # EXPAND ALL CONSTANTS TO PRIMITIVE VALUES (MVEL CAN ONLY ACCEPT PRIMITIVE VALUES)
    for c in constants:
        value = c.value
        n = c.name
        if len(split_field(n)) >= 3:
            continue    # DO NOT GO TOO DEEP
        if isinstance(value, list):
            continue  # DO NOT MESS WITH ARRAYS

        if isinstance(value, Mapping):
            for k, v in value.items():
                constants.append({"name": n + "." + k, "value": v})

    for c in reverse(constants):# REVERSE ORDER, SO LONGER NAMES ARE TESTED FIRST
        s = 0
        while True:
            s = expression.find(c.name, s)
            if s == -1:
                break
            if re.match(r"\w", expression[s - 1]):
                break
            if re.match(r"\w", expression[s + len(c.name)]):
                break

            v = value2MVEL(c.value)
            expression = expression[:s:] + "" + v + expression[:s + len(c.name):]

    return expression


def unpack_terms(facet, selects):
    # INTERPRET THE TERM-PACKED ES RESULTS AND RETURN DATA CUBE
    # ASSUME THE .term IS JSON OBJECT WITH ARRAY OF RESULT OBJECTS
    mod = len(selects)
    output = []
    for t in facet.terms:
        if t.term == "":
            continue        # NO DATA
        value = []
        for i, v in enumerate(t.term.split("|")):
            value.append(convert.pipe2value(v))
            if ((i + 1) % mod) == 0:
                value.append(t.count)
                output.append(value)
                value = []

    return output


#  PASS esFilter SIMPLIFIED ElasticSearch FILTER OBJECT
#  RETURN MVEL EXPRESSION
def _where(esFilter, _translate):
    if not esFilter or esFilter is True:
        return "true"

    keys = esFilter.keys()
    if len(keys) != 1:
        Log.error("Expecting only one filter aggregate")

    op = keys[0]
    if op == "and":
        list = esFilter[op]
        if not (list):
            return "true"
        if len(list) == 1:
            return _where(list[0], _translate)
        output = "(" + " && ".join(_where(l, _translate) for l in list) + ")"
        return output
    elif op == "or":
        list = esFilter[op]
        if not list:
            return "false"
        if len(list) == 1:
            return _where(list[0], _translate)
        output = "(" + " || ".join(_where(l, _translate) for l in list) + ")"
        return output
    elif op == "not":
        return "!(" + _where(esFilter[op, _translate]) + ")"
    elif op == "term":
        pair = esFilter[op]
        if len(pair.keys()) == 1:
            return [_translate(k) + "==" + value2MVEL(v) for k, v in pair.items()][0]
        else:
            return "(" + " && ".join(_translate(k) + "==" + value2MVEL(v) for k, v in pair.items()) + ")"
    elif op == "terms":
        output = []
        for variableName, valueList in esFilter[op].items():
            if not valueList:
                Log.error("Expecting something in 'terms' array")
            if len(valueList) == 1:
                output.append(_translate(variableName) + "==" + value2MVEL(valueList[0]))
            else:
                output.append("(" + " || ".join(_translate(variableName) + "==" + value2MVEL(v) for v in valueList) + ")")
        return " && ".join(output)
    elif op == "exists":
        # "exists":{"field":"myField"}
        pair = esFilter[op]
        variableName = pair.field
        return "(" + _translate(variableName) + "!=null)"
    elif op == "missing":
        fieldName = _translate(esFilter[op].field)
        testExistence = coalesce(esFilter[op].existence, True)
        testNull = coalesce(esFilter[op].null_value, True)

        output = []
        if testExistence and not testNull:
            output.append("(" + fieldName.replace(".?", ".") + " == empty)")        # REMOVE THE .? SO WE REFER TO THE FIELD, NOT GET THE VALUE
        if testNull:
            output.append("(" + fieldName + "==null)")
        return " || ".join(output)
    elif op == "range":
        pair = esFilter[op]
        ranges = []

        for variableName, r in pair.items():
            if r.gte:
                ranges.append(value2MVEL(r.gte) + "<=" + _translate(variableName))
            elif r.gt:
                ranges.append(value2MVEL(r.gt) + "<" + _translate(variableName))
            elif r["from"]:
                if r.include_lower == None or r.include_lower:
                    ranges.append(value2MVEL(r["from"]) + "<=" + _translate(variableName))
                else:
                    ranges.append(value2MVEL(r["from"]) + "<" + _translate(variableName))

            if r.lte:
                ranges.append(value2MVEL(r.lte) + ">=" + _translate(variableName))
            elif r.lt:
                ranges.append(value2MVEL(r.lt) + ">" + _translate(variableName))
            elif r["from"]:
                if r.include_lower == None or r.include_lower:
                    ranges.append(value2MVEL(r["from"]) + ">=" + _translate(variableName))
                else:
                    ranges.append(value2MVEL(r["from"]) + ">" + _translate(variableName))

        return "("+" && ".join(ranges)+")"

    elif op == "script":
        script = esFilter[op].script
        return _translate(script)
    elif op == "prefix":
        pair = esFilter[op]
        variableName, value = pair.items()[0]
        return _translate(variableName) + ".startsWith(" + quote(value) + ")"
    elif op == "match_all":
        return "true"
    else:
        Log.error("'" + op + "' is an unknown aggregate")

    return ""


VAR_CHAR = "abcdefghijklmnopqurstvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_.\""
keyword_pattern = re.compile(r"\.*\w*(?:\.\w+)*")


def isKeyword(value):
    """
    RETURN TRUE IF THE value IS JUST A NAME OF A FIELD, A LIST OF FIELDS, (OR A VALUE)
    """
    if not value or not isinstance(value, text_type):
        Log.error("Expecting a string")

    if keyword_pattern.match(value):
        return True
    return False


def value2MVEL(value):
    """
    FROM PYTHON VALUE TO MVEL EQUIVALENT
    """
    if isinstance(value, datetime):
        return str(convert.datetime2milli(value)) + " /*" + value.format("yyNNNdd HHmmss") + "*/"        # TIME
    if isinstance(value, Duration):
        return str(convert.timedelta2milli(value)) + " /*" + str(value) + "*/"    # DURATION

    if Math.is_number(value):
        return str(value)
    return quote(value)

# FROM PYTHON VALUE TO ES QUERY EQUIVALENT
def value2query(value):
    if isinstance(value, datetime):
        return convert.datetime2milli(value)
    if isinstance(value, Duration):
        return value.milli

    if Math.is_number(value):
        return value
    return quote(value)


def value2value(value):
    """
    CONVERT FROM PYTHON VALUE TO ES EQUIVALENT
    """
    if isinstance(value, datetime):
        return convert.datetime2milli(value)
    if isinstance(value, Duration):
        return value.milli    # DURATION
    return value







def addFunctions(mvel):
    """
    PREPEND THE REQUIRED MVEL FUNCTIONS TO THE CODE
    """
    isAdded = Data()            # SOME FUNCTIONS DEPEND ON OTHERS

    head=[]
    body=mvel

    keepAdding = True
    while keepAdding:
        keepAdding = False
        for func_name, func_code in FUNCTIONS.items():
            if isAdded[func_name]:
                continue
            if mvel.find(func_name) == -1:
                continue
            keepAdding = True
            isAdded[func_name] = func_code
            head.append(func_code)
            mvel = func_code + mvel
    return Data(
        head="".join(head),
        body=body
    )


FUNCTIONS = {
    "String2Quote":
        "var String2Quote = function(str){\n" +
        "if (!(str is String)){ str; }else{\n" + # LAST VALUE IS RETURNED.  "return" STOPS EXECUTION COMPLETELY!
        "" + value2MVEL("\"") + "+" +
        "str.replace(" + value2MVEL("\\") + "," + value2MVEL("\\\\") +
        ").replace(" + value2MVEL("\"") + "," + value2MVEL("\\\"") +
        ").replace(" + value2MVEL("\'") + "," + value2MVEL("\\\'") + ")+" +
        value2MVEL("\"") + ";\n" +
        "}};\n",

    "Value2Pipe":
        'var Value2Pipe = function(value){\n' + # SPACES ARE IMPORTANT BETWEEN "="
        "if (value==null){ \"0\" }else " +
        "if (value is ArrayList || value is org.elasticsearch.common.mvel2.util.FastList){" +
        "var out = \"\";\n" +
        "foreach (v : value) out = (out==\"\") ? v : out + \"|\" + Value2Pipe(v);\n" +
        "'a'+Value2Pipe(out);\n" +
        "}else \n" +
        "if (value is Long || value is Integer || value is Double){ 'n'+value; }else \n" +
        "if (!(value is String)){ 's'+value.getClass().getName(); }else \n" +
        '"s"+value.replace("\\\\", "\\\\\\\\").replace("|", "\\\\p");' + # CAN NOT value TO MAKE NUMBER A STRING (OR EVEN TO PREPEND A STRING!)
        "};\n",

    # 	"replaceAll":
    # 		"var replaceAll = function(output, find, replace){\n" +
    # 			"if (output.length()==0) return output;\n"+
    # 			"s = output.indexOf(find, 0);\n" +
    # 			"while(s>=0){\n" +
    # 				"output=output.replace(find, replace);\n" +
    # 				"s=s-find.length()+replace.length();\n" +
    # 				"s = output.indexOf(find, s);\n" +
    # 			"}\n"+
    # 			"output;\n"+
    # 		'};\n',

    "floorDay":
        "var floorDay = function(value){ Math.floor(value/86400000))*86400000;};\n",

    "floorInterval":
        "var floorInterval = function(value, interval){ Math.floor((double)value/(double)interval)*interval;};\n",

    "maximum": # JUST BECAUSE MVEL'S MAX ONLY USES MAX(int, int).  G*DDA*NIT!
        "var maximum = function(a, b){if (a==null) b; else if (b==null) a; else if (a>b) a; else b;\n};\n",

    "minimum": # JUST BECAUSE MVEL'S MAX ONLY USES MAX(int, int).  G*DDA*NIT!
        "var minimum = function(a, b){if (a==null) b; else if (b==null) a; else if (a<b) a; else b;\n};\n",

    "coalesce": # PICK FIRST NOT-NULL VALUE
        "var coalesce = function(a, b){if (a==null) b; else a; \n};\n",

    "zero2null": # ES MAKES IT DIFFICULT TO DETECT NULL/MISSING VALUES, BUT WHEN DEALING WITH NUMBERS, ES DEFAULTS TO RETURNING ZERO FOR missing VALUES!!
        "var zero2null = function(a){if (a==0) null; else a; \n};\n",

    "get": # MY OWN PERSONAL *FU* TO THE TWISTED MVEL PROPERTY ACCESS
        "var get = function(hash, key){\n" +
        "if (hash==null) null; else hash[key];\n" +
        "};\n",

    "isNumeric":
        "var isNumeric = function(value){\n" +
        "value = value + \"\";\n" +
        # 			"try{ value-0; }catch(e){ 0; }"+
        "var isNum = value.length()>0;\n" +
        "for (v : value.toCharArray()){\n" +
        "if (\"0123456789\".indexOf(v)==-1) isNum = false;\n" +
        "};\n" +
        "isNum;\n" +
        "};\n",

    "alpha2zero":
        "var alpha2zero = function(value){\n" +
        "var output = 0;\n" +
        "if (isNumeric(value)) output = value-0;\n" +
        "return output;" +
        "};\n",

    # KANBAN SOFTWARE
    # CAN SEE QUEUE BLOCKAGES AND SEE SINGLE BLOCKERS


    "concat":
        "var concat = function(array){\n" +
        "if (array==null) \"\"; else {\n" +
        "var output = \"\";\n" +
        "for (v : array){ output = output+\"|\"+v+\"|\"; };\n" +
        "output;\n" +
        "}};\n",

    # 	"contains":
    # 		"var contains = function(array, value){\n"+
    # 			"if (array==null) false; else {\n"+
    # 			"var good = false;\n"+
    # 			"for (v : array){ if (v==value) good=true; };\n"+
    # 			"good;\n"+
    # 		"}};\n",

    "getFlagValue": # SPECIFICALLY FOR cf_* FLAGS: CONCATENATE THE ATTRIBUTE NAME WITH ATTRIBUTE VALUE, IF EXISTS
        "var getFlagValue = function(name){\n" +
        "if (_source[name]!=null)" +
        "\" \"+name+_source[name];\n" +
        "else \n" +
        "\"\";\n" +
        "};\n",

    "getDocValue":
        "var getDocValue = function(name){\n" +
        "var out = [];\n" +
        "var v = doc[name];\n" +
        # 			"if (v is org.elasticsearch.common.mvel2.ast.Function) v = v();=n" +
        "if (v==null || v.value==null) { null; } else\n" +
        "if (v.values.size()<=1){ v.value; } else\n" + # ES MAKES NO DISTINCTION BETWEEN v or [v], SO NEITHER DO I
        "{for(k : v.values) out.add(k); out;}" +
        "};\n",

    "getSourceValue":
        "var getSourceValue = function(name){\n" +
        "var out = [];\n" +
        "var v = _source[name];\n" +
        # 			"if (v is org.elasticsearch.common.mvel2.ast.Function) v = v();=n" +
        "if (v==null) { null; } else\n" +
        "if (v[\"values\"]==null || v.values.size()<=1){ v.value; } else {\n" + # ES MAKES NO DISTINCTION BETWEEN v or [v], SO NEITHER DO I
        "for(k : v) out.add(k); out;\n" + # .size() MUST BE USED INSTEAD OF .length, THE LATTER WILL CRASH IF JITTED (https://github.com/elasticsearch/elasticsearch/issues/3094)
        "}};\n",

    "getDocArray":
        "var getDocArray = function(name){\n" +
        "var out = [];\n" +
        "var v = doc[name];\n" +
        "if (v!=null && v.value!=null) for(k : v.values) out.add(k);" +
        "out;" +
        "};\n",


    "milli2Month":
        "var milli2Month = function(value, milliOffset){\n" +
        "g=new java.util.GregorianCalendar(new java.util.SimpleTimeZone(0, \"GMT\"));\n" +
        "g.setTimeInMillis(value);\n" +
        "g.add(java.util.GregorianCalendar.MILLISECOND, -milliOffset);\n" +
        "m = g.get(java.util.GregorianCalendar.MONTH);\n" +
        "output = \"\"+g.get(java.util.GregorianCalendar.YEAR)+(m>9?\"\":\"0\")+m;\n" +
        "output;\n" +
        "};\n",

    "between":
        "var between = function(value, prefix, suffix){\n" +
        "if (value==null){ null; }else{\n" +
        "var start = value.indexOf(prefix, 0);\n" +
        "if (start==-1){ null; }else{\n" +
        "var end = value.indexOf(suffix, start+prefix.length());\n" +
        "if (end==-1){  null; }else{\n" +
        "value.substring(start+prefix.length(), end);\n" +
        "}}}\n" +
        "};\n"
}


def replacePrefix(value, prefix, new_prefix):
    try:
        if value.startswith(prefix):
            return new_prefix+value[len(prefix)::]
        return value
    except Exception as e:
        Log.error("can not replace prefix", e)

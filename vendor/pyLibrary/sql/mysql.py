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

import subprocess
from collections import Mapping
from datetime import datetime

from pymysql import connect, InterfaceError, cursors

import mo_json
from jx_python import jx
from mo_dots import coalesce, wrap, listwrap, unwrap
from mo_files import File
from mo_future import text_type, utf8_json_encoder
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import Except, suppress_exception
from mo_logs.strings import expand_template
from mo_logs.strings import indent
from mo_logs.strings import outdent
from mo_math import Math
from mo_times import Date
from pyLibrary.sql import SQL

DEBUG = False
MAX_BATCH_SIZE = 100
EXECUTE_TIMEOUT = 5*600*1000  # in milliseconds


all_db = []


class MySQL(object):
    """
    Parameterize SQL by name rather than by position.  Return records as objects
    rather than tuples.
    """
    @override
    def __init__(
        self,
        host,
        username,
        password,
        port=3306,
        debug=False,
        schema=None,
        preamble=None,
        readonly=False,
        kwargs=None
    ):
        """
        OVERRIDE THE settings.schema WITH THE schema PARAMETER
        preamble WILL BE USED TO ADD COMMENTS TO THE BEGINNING OF ALL SQL
        THE INTENT IS TO HELP ADMINISTRATORS ID THE SQL RUNNING ON THE DATABASE

        schema - NAME OF DEFAULT database/schema IN QUERIES

        preamble - A COMMENT TO BE ADDED TO EVERY SQL STATEMENT SENT

        readonly - USED ONLY TO INDICATE IF A TRANSACTION WILL BE OPENED UPON
        USE IN with CLAUSE, YOU CAN STILL SEND UPDATES, BUT MUST OPEN A
        TRANSACTION BEFORE YOU DO
        """
        all_db.append(self)

        self.settings = kwargs
        self.cursor = None
        self.query_cursor = None
        if preamble == None:
            self.preamble = ""
        else:
            self.preamble = indent(preamble, "# ").strip() + "\n"

        self.readonly = readonly
        self.debug = coalesce(debug, DEBUG)
        if host:
            self._open()

    def _open(self):
        """ DO NOT USE THIS UNLESS YOU close() FIRST"""
        try:
            self.db = connect(
                host=self.settings.host,
                port=self.settings.port,
                user=coalesce(self.settings.username, self.settings.user),
                passwd=coalesce(self.settings.password, self.settings.passwd),
                db=coalesce(self.settings.schema, self.settings.db),
                read_timeout=coalesce(self.settings.read_timeout, (EXECUTE_TIMEOUT/1000)-10),
                charset=u"utf8",
                use_unicode=True,
                ssl=coalesce(self.settings.ssl, None),
                cursorclass=cursors.SSCursor
            )
        except Exception as e:
            if self.settings.host.find("://") == -1:
                Log.error(u"Failure to connect to {{host}}:{{port}}",
                    host= self.settings.host,
                    port= self.settings.port,
                    cause=e
                )
            else:
                Log.error(u"Failure to connect.  PROTOCOL PREFIX IS PROBABLY BAD", e)
        self.cursor = None
        self.partial_rollback = False
        self.transaction_level = 0
        self.backlog = []     # accumulate the write commands so they are sent at once
        if self.readonly:
            self.begin()

    def __enter__(self):
        if not self.readonly:
            self.begin()
        return self

    def __exit__(self, type, value, traceback):
        if self.readonly:
            self.close()
            return

        if isinstance(value, BaseException):
            try:
                if self.cursor: self.cursor.close()
                self.cursor = None
                self.rollback()
            except Exception as e:
                Log.warning(u"can not rollback()", cause=[value, e])
            finally:
                self.close()
            return

        try:
            self.commit()
        except Exception as e:
            Log.warning(u"can not commit()", e)
        finally:
            self.close()


    def transaction(self):
        """
        return not-started transaction (for with statement)
        """
        return Transaction(self)

    def begin(self):
        if self.transaction_level == 0:
            self.cursor = self.db.cursor()
        self.transaction_level += 1
        self.execute("SET TIME_ZONE='+00:00'")
        self.execute("SET MAX_EXECUTION_TIME=" + text_type(EXECUTE_TIMEOUT))


    def close(self):
        if self.transaction_level > 0:
            if self.readonly:
                self.commit()  # AUTO-COMMIT
            else:
                Log.error("expecting commit() or rollback() before close")
        self.cursor = None  # NOT NEEDED
        try:
            self.db.close()
        except Exception as e:
            if e.message.find("Already closed") >= 0:
                return

            Log.warning("can not close()", e)
        finally:
            try:
                all_db.remove(self)
            except Exception as e:
                Log.error("not expected", cause=e)

    def commit(self):
        try:
            self._execute_backlog()
        except Exception as e:
            with suppress_exception:
                self.rollback()
            Log.error("Error while processing backlog", e)

        if self.transaction_level == 0:
            Log.error("No transaction has begun")
        elif self.transaction_level == 1:
            if self.partial_rollback:
                with suppress_exception:
                    self.rollback()

                Log.error("Commit after nested rollback is not allowed")
            else:
                if self.cursor:
                    self.cursor.close()
                self.cursor = None
                self.db.commit()

        self.transaction_level -= 1

    def flush(self):
        try:
            self.commit()
        except Exception as e:
            Log.error("Can not flush", e)

        try:
            self.begin()
        except Exception as e:
            Log.error("Can not flush", e)

    def rollback(self):
        self.backlog = []     # YAY! FREE!
        if self.transaction_level == 0:
            Log.error("No transaction has begun")
        elif self.transaction_level == 1:
            self.transaction_level -= 1
            if self.cursor != None:
                self.cursor.close()
            self.cursor = None
            self.db.rollback()
        else:
            self.transaction_level -= 1
            self.partial_rollback = True
            Log.warning("Can not perform partial rollback!")

    def call(self, proc_name, params):
        self._execute_backlog()
        params = [unwrap(v) for v in params]
        try:
            self.cursor.callproc(proc_name, params)
            self.cursor.close()
            self.cursor = self.db.cursor()
        except Exception as e:
            Log.error("Problem calling procedure " + proc_name, e)

    def query(self, sql, param=None, stream=False, row_tuples=False):
        """
        RETURN LIST OF dicts
        """
        if not self.cursor:  # ALLOW NON-TRANSACTIONAL READS
            Log.error("must perform all queries inside a transaction")
        self._execute_backlog()

        try:
            if param:
                sql = expand_template(sql, self.quote_param(param))
            sql = self.preamble + outdent(sql)
            if self.debug:
                Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))

            self.cursor.execute(sql)
            if row_tuples:
                if stream:
                    result = self.cursor
                else:
                    result = wrap(list(self.cursor))
            else:
                columns = [utf8_to_unicode(d[0]) for d in coalesce(self.cursor.description, [])]
                if stream:
                    result = (wrap({c: utf8_to_unicode(v) for c, v in zip(columns, row)}) for row in self.cursor)
                else:
                    result = wrap([{c: utf8_to_unicode(v) for c, v in zip(columns, row)} for row in self.cursor])

            return result
        except Exception as e:
            if isinstance(e, InterfaceError) or e.message.find("InterfaceError") >= 0:
                Log.error("Did you close the db connection?", e)
            Log.error("Problem executing SQL:\n{{sql|indent}}",  sql= sql, cause=e, stack_depth=1)

    def column_query(self, sql, param=None):
        """
        RETURN RESULTS IN [column][row_num] GRID
        """
        self._execute_backlog()
        try:
            old_cursor = self.cursor
            if not old_cursor: # ALLOW NON-TRANSACTIONAL READS
                self.cursor = self.db.cursor()
                self.cursor.execute("SET TIME_ZONE='+00:00'")
                self.cursor.close()
                self.cursor = self.db.cursor()

            if param:
                sql = expand_template(sql, self.quote_param(param))
            sql = self.preamble + outdent(sql)
            if self.debug:
                Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))

            self.cursor.execute(sql)
            grid = [[utf8_to_unicode(c) for c in row] for row in self.cursor]
            # columns = [utf8_to_unicode(d[0]) for d in coalesce(self.cursor.description, [])]
            result = zip(*grid)

            if not old_cursor:   # CLEANUP AFTER NON-TRANSACTIONAL READS
                self.cursor.close()
                self.cursor = None

            return result
        except Exception as e:
            if isinstance(e, InterfaceError) or e.message.find("InterfaceError") >= 0:
                Log.error("Did you close the db connection?", e)
            Log.error("Problem executing SQL:\n{{sql|indent}}",  sql= sql, cause=e,stack_depth=1)




    # EXECUTE GIVEN METHOD FOR ALL ROWS RETURNED
    def forall(self, sql, param=None, _execute=None):
        assert _execute
        num = 0

        self._execute_backlog()
        try:
            old_cursor = self.cursor
            if not old_cursor: # ALLOW NON-TRANSACTIONAL READS
                self.cursor = self.db.cursor()

            if param:
                sql = expand_template(sql, self.quote_param(param))
            sql = self.preamble + outdent(sql)
            if self.debug:
                Log.note("Execute SQL:\n{{sql}}",  sql=indent(sql))
            self.cursor.execute(sql)

            columns = tuple([utf8_to_unicode(d[0]) for d in self.cursor.description])
            for r in self.cursor:
                num += 1
                _execute(wrap(dict(zip(columns, [utf8_to_unicode(c) for c in r]))))

            if not old_cursor:   # CLEANUP AFTER NON-TRANSACTIONAL READS
                self.cursor.close()
                self.cursor = None

        except Exception as e:
            Log.error("Problem executing SQL:\n{{sql|indent}}",  sql= sql, cause=e, stack_depth=1)

        return num


    def execute(self, sql, param=None):
        if self.transaction_level == 0:
            Log.error("Expecting transaction to be started before issuing queries")

        if param:
            sql = expand_template(sql, self.quote_param(param))
        sql = outdent(sql)
        self.backlog.append(sql)
        if self.debug or len(self.backlog) >= MAX_BATCH_SIZE:
            self._execute_backlog()


    @staticmethod
    @override
    def execute_sql(
        host,
        username,
        password,
        sql,
        schema=None,
        param=None,
        kwargs=None
    ):
        """EXECUTE MANY LINES OF SQL (FROM SQLDUMP FILE, MAYBE?"""
        kwargs.schema = coalesce(kwargs.schema, kwargs.database)

        if param:
            with MySQL(kwargs) as temp:
                sql = expand_template(sql, temp.quote_param(param))

        # MWe have no way to execute an entire SQL file in bulk, so we
        # have to shell out to the commandline client.
        args = [
            "mysql",
            "-h{0}".format(kwargs.host),
            "-u{0}".format(kwargs.username),
            "-p{0}".format(kwargs.password)
        ]
        if kwargs.schema:
            args.append("{0}".format(kwargs.schema))

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=-1
            )
            if isinstance(sql, text_type):
                sql = sql.encode("utf8")
            (output, _) = proc.communicate(sql)
        except Exception as e:
            raise Log.error("Can not call \"mysql\"", e)

        if proc.returncode:
            if len(sql) > 10000:
                sql = "<" + text_type(len(sql)) + " bytes of sql>"
            Log.error("Unable to execute sql: return code {{return_code}}, {{output}}:\n {{sql}}\n",
                sql=indent(sql),
                return_code=proc.returncode,
                output=output
            )

    @staticmethod
    @override
    def execute_file(
        filename,
        host,
        username,
        password,
        schema=None,
        param=None,
        ignore_errors=False,
        kwargs=None
    ):
        # MySQLdb provides no way to execute an entire SQL file in bulk, so we
        # have to shell out to the commandline client.
        sql = File(filename).read()
        if ignore_errors:
            with suppress_exception:
                MySQL.execute_sql(sql=sql, param=param, kwargs=kwargs)
        else:
            MySQL.execute_sql(sql=sql, param=param, kwargs=kwargs)

    def _execute_backlog(self):
        if not self.backlog: return

        (backlog, self.backlog) = (self.backlog, [])
        if self.db.__module__.startswith("pymysql"):
            # BUG IN PYMYSQL: CAN NOT HANDLE MULTIPLE STATEMENTS
            # https://github.com/PyMySQL/PyMySQL/issues/157
            for b in backlog:
                sql = self.preamble + b
                try:
                    if self.debug:
                        Log.note("Execute SQL:\n{{sql|indent}}",  sql= sql)
                    self.cursor.execute(b)
                except Exception as e:
                    Log.error("Can not execute sql:\n{{sql}}",  sql= sql, cause=e)

            self.cursor.close()
            self.cursor = self.db.cursor()
        else:
            for i, g in jx.groupby(backlog, size=MAX_BATCH_SIZE):
                sql = self.preamble + ";\n".join(g)
                try:
                    if self.debug:
                        Log.note("Execute block of SQL:\n{{sql|indent}}",  sql= sql)
                    self.cursor.execute(sql)
                    self.cursor.close()
                    self.cursor = self.db.cursor()
                except Exception as e:
                    Log.error("Problem executing SQL:\n{{sql|indent}}",  sql= sql, cause=e, stack_depth=1)


    ## Insert dictionary of values into table
    def insert(self, table_name, record):
        keys = record.keys()

        try:
            command = (
                "INSERT INTO " + self.quote_column(table_name) + "(" +
                SQL(",").join([self.quote_column(k) for k in keys]) +
                ") VALUES (" +
                SQL(",").join([self.quote_value(record[k]) for k in keys]) +
                ")"
            )
            self.execute(command)
        except Exception as e:
            Log.error("problem with record: {{record}}",  record= record, cause=e)

    # candidate_key IS LIST OF COLUMNS THAT CAN BE USED AS UID (USUALLY PRIMARY KEY)
    # ONLY INSERT IF THE candidate_key DOES NOT EXIST YET
    def insert_new(self, table_name, candidate_key, new_record):
        candidate_key = listwrap(candidate_key)

        condition = " AND\n".join([self.quote_column(k) + "=" + self.quote_value(new_record[k]) if new_record[k] != None else self.quote_column(k) + " IS Null" for k in candidate_key])
        command = (
            "INSERT INTO " + self.quote_column(table_name) + " (" +
            ",".join([self.quote_column(k) for k in new_record.keys()]) +
            ")\n" +
            "SELECT a.* FROM (SELECT " + ",".join([self.quote_value(v) + " " + self.quote_column(k) for k, v in new_record.items()]) + " FROM DUAL) a\n" +
            "LEFT JOIN " +
            "(SELECT 'dummy' exist FROM " + self.quote_column(table_name) + " WHERE " + condition + " LIMIT 1) b ON 1=1 WHERE exist IS Null"
        )
        self.execute(command, {})


    # ONLY INSERT IF THE candidate_key DOES NOT EXIST YET
    def insert_newlist(self, table_name, candidate_key, new_records):
        for r in new_records:
            self.insert_new(table_name, candidate_key, r)


    def insert_list(self, table_name, records):
        if not records:
            return

        keys = set()
        for r in records:
            keys |= set(r.keys())
        keys = jx.sort(keys)

        try:
            command = (
                "INSERT INTO " + self.quote_column(table_name) + "(" +
                ",".join([self.quote_column(k) for k in keys]) +
                ") VALUES " + ",\n".join([
                    "(" + ",".join([self.quote_value(r[k]) for k in keys]) + ")"
                    for r in records
                ])
            )
            self.execute(command)
        except Exception as e:
            Log.error("problem with record: {{record}}",  record= records, cause=e)


    def update(self, table_name, where_slice, new_values):
        """
        where_slice - A Data WHICH WILL BE USED TO MATCH ALL IN table
                      eg {"id": 42}
        new_values  - A dict WITH COLUMN NAME, COLUMN VALUE PAIRS TO SET
        """
        new_values = self.quote_param(new_values)

        where_clause = " AND\n".join([
            self.quote_column(k) + "=" + self.quote_value(v) if v != None else self.quote_column(k) + " IS NULL"
            for k, v in where_slice.items()
        ])

        command = (
            "UPDATE " + self.quote_column(table_name) + "\n" +
            "SET " +
            ",\n".join([self.quote_column(k) + "=" + v for k, v in new_values.items()]) + "\n" +
            "WHERE " +
            where_clause
        )
        self.execute(command, {})


    def quote_param(self, param):
        return {k: self.quote_value(v) for k, v in param.items()}

    def quote_value(self, value):
        """
        convert values to mysql code for the same
        mostly delegate directly to the mysql lib, but some exceptions exist
        """
        try:
            if value == None:
                return SQL("NULL")
            elif isinstance(value, SQL):
                if not value.param:
                    # value.template CAN BE MORE THAN A TEMPLATE STRING
                    return self.quote_sql(value.template)
                param = {k: self.quote_sql(v) for k, v in value.param.items()}
                return SQL(expand_template(value.template, param))
            elif isinstance(value, text_type):
                return SQL(self.db.literal(value))
            elif isinstance(value, Mapping):
                return SQL(self.db.literal(json_encode(value)))
            elif Math.is_number(value):
                return SQL(text_type(value))
            elif isinstance(value, datetime):
                return SQL("str_to_date('" + value.strftime("%Y%m%d%H%M%S.%f") + "', '%Y%m%d%H%i%s.%f')")
            elif isinstance(value, Date):
                return SQL("str_to_date('"+value.format("%Y%m%d%H%M%S.%f")+"', '%Y%m%d%H%i%s.%f')")
            elif hasattr(value, '__iter__'):
                return SQL(self.db.literal(json_encode(value)))
            else:
                return self.db.literal(value)
        except Exception as e:
            Log.error("problem quoting SQL", e)


    def quote_sql(self, value, param=None):
        """
        USED TO EXPAND THE PARAMETERS TO THE SQL() OBJECT
        """
        try:
            if isinstance(value, SQL):
                if not param:
                    return value
                param = {k: self.quote_sql(v) for k, v in param.items()}
                return expand_template(value, param)
            elif isinstance(value, text_type):
                return value
            elif isinstance(value, Mapping):
                return self.db.literal(json_encode(value))
            elif hasattr(value, '__iter__'):
                return "(" + ",".join([self.quote_sql(vv) for vv in value]) + ")"
            else:
                return text_type(value)
        except Exception as e:
            Log.error("problem quoting SQL", e)

    def quote_column(self, column_name, table=None):
        if column_name==None:
            Log.error("missing column_name")
        elif isinstance(column_name, text_type):
            if table:
                column_name = table + "." + column_name
            return SQL("`" + column_name.replace(".", "`.`") + "`")    # MY SQL QUOTE OF COLUMN NAMES
        elif isinstance(column_name, list):
            if table:
                return SQL(", ".join([self.quote_column(table + "." + c) for c in column_name]))
            return SQL(", ".join([self.quote_column(c) for c in column_name]))
        else:
            # ASSUME {"name":name, "value":value} FORM
            return SQL(column_name.value + " AS " + self.quote_column(column_name.name))

    def sort2sqlorderby(self, sort):
        sort = jx.normalize_sort_parameters(sort)
        return ",\n".join([self.quote_column(s.field) + (" DESC" if s.sort == -1 else " ASC") for s in sort])


def utf8_to_unicode(v):
    try:
        if isinstance(v, str):
            return v.decode("utf8")
        else:
            return v
    except Exception as e:
        Log.error("not expected", e)




def int_list_packer(term, values):
    """
    return singletons, ranges and exclusions
    """
    DENSITY = 10  # a range can have holes, this is inverse of the hole density
    MIN_RANGE = 20  # min members before a range is allowed to be used

    singletons = set()
    ranges = []
    exclude = set()

    sorted = jx.sort(values)

    last = sorted[0]
    curr_start = last
    curr_excl = set()

    for v in sorted[1::]:
        if v <= last + 1:
            pass
        elif v - last > 3:
            # big step, how do we deal with it?
            if last == curr_start:
                # not a range yet, so just add as singlton
                singletons.add(last)
            elif last - curr_start - len(curr_excl) < MIN_RANGE or ((last - curr_start) < len(curr_excl) * DENSITY):
                # small ranges are singletons, sparse ranges are singletons
                singletons |= set(range(curr_start, last + 1))
                singletons -= curr_excl
            else:
                # big enough, and dense enough range
                ranges.append({"gte": curr_start, "lte": last})
                exclude |= curr_excl
            curr_start = v
            curr_excl = set()
        else:
            if 1 + last - curr_start >= len(curr_excl) * DENSITY:
                # high density, keep track of excluded and continue
                add_me = set(range(last + 1, v))
                curr_excl |= add_me
            elif 1 + last - curr_start - len(curr_excl) < MIN_RANGE:
                # not big enough, convert range to singletons
                new_singles = set(range(curr_start, last + 1)) - curr_excl
                singletons = singletons | new_singles

                curr_start = v
                curr_excl = set()
            else:
                ranges.append({"gte": curr_start, "lte": last})
                exclude |= curr_excl
                curr_start = v
                curr_excl = set()
        last = v

    if last == curr_start:
        # not a range yet, so just add as singlton
        singletons.add(last)
    elif last - curr_start - len(curr_excl) < MIN_RANGE or ((last - curr_start) < len(curr_excl) * DENSITY):
        # small ranges are singletons, sparse ranges are singletons
        singletons |= set(range(curr_start, last + 1))
        singletons -= curr_excl
    else:
        # big enough, and dense enough range
        ranges.append({"gte": curr_start, "lte": last})
        exclude |= curr_excl

    if ranges:
        r = {"or": [{"range": {term: r}} for r in ranges]}
        if exclude:
            r = {"and": [r, {"not": {"terms": {term: jx.sort(exclude)}}}]}
        if singletons:
            return {"or": [
                {"terms": {term: jx.sort(singletons)}},
                r
            ]}
        else:
            return r
    else:
        raise Except("no packing possible")


class Transaction(object):
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, Exception):
            self.db.rollback()
        else:
            self.db.commit()


def json_encode(value):
    """
    FOR PUTTING JSON INTO DATABASE (sort_keys=True)
    dicts CAN BE USED AS KEYS
    """
    return text_type(utf8_json_encoder(mo_json.scrub(value)))


mysql_type_to_json_type = {
    "bigint": "number",
    "blob": "string",
    "char": "string",
    "datetime": "number",
    "decimal": "number",
    "double": "number",
    "enum": "number",
    "float": "number",
    "int": "number",
    "longblob": "string",
    "longtext": "string",
    "mediumblob": "string",
    "mediumint": "number",
    "mediumtext": "string",
    "set": "array",
    "smallint": "number",
    "text": "string",
    "time": "number",
    "timestamp": "number",
    "tinyint": "number",
    "tinytext": "number",
    "varchar": "string"
}

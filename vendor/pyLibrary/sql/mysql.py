# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

from datetime import datetime
import subprocess

from pymysql import InterfaceError, connect, cursors

from jx_python import jx
from mo_dots import coalesce, is_data, is_list, listwrap, split_field, unwrap, wrap
from mo_files import File
from mo_future import is_binary, is_text, text_type, transpose, utf8_json_encoder
import mo_json
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import Except, suppress_exception
from mo_logs.strings import expand_template, indent, outdent
from mo_math import is_number
from mo_times import Date
from pyLibrary.sql import SQL, SQL_AND, SQL_ASC, SQL_DESC, SQL_FROM, SQL_IS_NULL, SQL_LEFT_JOIN, SQL_LIMIT, SQL_NULL, SQL_ONE, SQL_SELECT, SQL_TRUE, SQL_WHERE, sql_alias, sql_iso, sql_list
from pyLibrary.sql.sqlite import join_column

DEBUG = False
MAX_BATCH_SIZE = 100
EXECUTE_TIMEOUT = 5 * 600 * 1000  # in milliseconds  SET TO ZERO (OR None) FOR HOST DEFAULT TIMEOUT

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
                read_timeout=coalesce(self.settings.read_timeout, (EXECUTE_TIMEOUT / 1000) - 10 if EXECUTE_TIMEOUT else None, 5*60),
                charset=u"utf8",
                use_unicode=True,
                ssl=coalesce(self.settings.ssl, None),
                cursorclass=cursors.SSCursor
            )
        except Exception as e:
            if self.settings.host.find("://") == -1:
                Log.error(
                    u"Failure to connect to {{host}}:{{port}}",
                    host=self.settings.host,
                    port=self.settings.port,
                    cause=e
                )
            else:
                Log.error(u"Failure to connect.  PROTOCOL PREFIX IS PROBABLY BAD", e)
        self.cursor = None
        self.partial_rollback = False
        self.transaction_level = 0
        self.backlog = []  # accumulate the write commands so they are sent at once
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
        if EXECUTE_TIMEOUT:
            try:
                self.execute("SET MAX_EXECUTION_TIME=" + text_type(EXECUTE_TIMEOUT))
                self._execute_backlog()
            except Exception as e:
                e = Except.wrap(e)
                if "Unknown system variable 'MAX_EXECUTION_TIME'" in e:
                    globals()['EXECUTE_TIMEOUT'] = 0  # THIS VERSION OF MYSQL DOES NOT HAVE SESSION LEVEL VARIABLE
                else:
                    raise e

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
            e = Except.wrap(e)
            if "Already closed" in e:
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
        self.backlog = []  # YAY! FREE!
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
                sql = expand_template(sql, quote_param(param))
            sql = self.preamble + outdent(sql)
            self.debug and Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))

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
            e = Except.wrap(e)
            if "InterfaceError" in e:
                Log.error("Did you close the db connection?", e)
            Log.error("Problem executing SQL:\n{{sql|indent}}", sql=sql, cause=e, stack_depth=1)

    def column_query(self, sql, param=None):
        """
        RETURN RESULTS IN [column][row_num] GRID
        """
        self._execute_backlog()
        try:
            old_cursor = self.cursor
            if not old_cursor:  # ALLOW NON-TRANSACTIONAL READS
                self.cursor = self.db.cursor()
                self.cursor.execute("SET TIME_ZONE='+00:00'")
                self.cursor.close()
                self.cursor = self.db.cursor()

            if param:
                sql = expand_template(sql, quote_param(param))
            sql = self.preamble + outdent(sql)
            self.debug and Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))

            self.cursor.execute(sql)
            grid = [[utf8_to_unicode(c) for c in row] for row in self.cursor]
            # columns = [utf8_to_unicode(d[0]) for d in coalesce(self.cursor.description, [])]
            result = transpose(*grid)

            if not old_cursor:  # CLEANUP AFTER NON-TRANSACTIONAL READS
                self.cursor.close()
                self.cursor = None

            return result
        except Exception as e:
            if isinstance(e, InterfaceError) or e.message.find("InterfaceError") >= 0:
                Log.error("Did you close the db connection?", e)
            Log.error("Problem executing SQL:\n{{sql|indent}}", sql=sql, cause=e, stack_depth=1)

    # EXECUTE GIVEN METHOD FOR ALL ROWS RETURNED
    def forall(self, sql, param=None, _execute=None):
        assert _execute
        num = 0

        self._execute_backlog()
        try:
            old_cursor = self.cursor
            if not old_cursor:  # ALLOW NON-TRANSACTIONAL READS
                self.cursor = self.db.cursor()

            if param:
                sql = expand_template(sql, quote_param(param))
            sql = self.preamble + outdent(sql)
            self.debug and Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))
            self.cursor.execute(sql)

            columns = tuple([utf8_to_unicode(d[0]) for d in self.cursor.description])
            for r in self.cursor:
                num += 1
                _execute(wrap(dict(zip(columns, [utf8_to_unicode(c) for c in r]))))

            if not old_cursor:  # CLEANUP AFTER NON-TRANSACTIONAL READS
                self.cursor.close()
                self.cursor = None

        except Exception as e:
            Log.error("Problem executing SQL:\n{{sql|indent}}", sql=sql, cause=e, stack_depth=1)

        return num

    def execute(self, sql, param=None):
        if self.transaction_level == 0:
            Log.error("Expecting transaction to be started before issuing queries")

        if param:
            sql = expand_template(sql, quote_param(param))
        sql = outdent(sql)
        self.backlog.append(sql)
        if self.debug or len(self.backlog) >= MAX_BATCH_SIZE:
            self._execute_backlog()

    def _execute_backlog(self):
        if not self.backlog: return

        backlog, self.backlog = self.backlog, []
        for i, g in jx.groupby(backlog, size=MAX_BATCH_SIZE):
            sql = self.preamble + ";\n".join(g)
            try:
                self.debug and Log.note("Execute block of SQL:\n{{sql|indent}}", sql=sql)
                self.cursor.execute(sql)
                self.cursor.close()
                self.cursor = self.db.cursor()
            except Exception as e:
                Log.error("Problem executing SQL:\n{{sql|indent}}", sql=sql, cause=e, stack_depth=1)

    ## Insert dictionary of values into table
    def insert(self, table_name, record):
        keys = list(record.keys())

        try:
            command = (
                "INSERT INTO " + quote_column(table_name) +
                sql_iso(sql_list([quote_column(k) for k in keys])) +
                " VALUES " +
                sql_iso(sql_list([quote_value(record[k]) for k in keys]))
            )
            self.execute(command)
        except Exception as e:
            Log.error("problem with record: {{record}}", record=record, cause=e)

    # candidate_key IS LIST OF COLUMNS THAT CAN BE USED AS UID (USUALLY PRIMARY KEY)
    # ONLY INSERT IF THE candidate_key DOES NOT EXIST YET
    def insert_new(self, table_name, candidate_key, new_record):
        candidate_key = listwrap(candidate_key)

        condition = SQL_AND.join([
            quote_column(k) + "=" + quote_value(new_record[k])
            if new_record[k] != None
            else quote_column(k) + SQL_IS_NULL
            for k in candidate_key
        ])
        command = (
            "INSERT INTO " + quote_column(table_name) + sql_iso(sql_list(
                quote_column(k) for k in new_record.keys()
            )) +
            SQL_SELECT + "a.*" + SQL_FROM + sql_iso(
                SQL_SELECT + sql_list([quote_value(v) + " " + quote_column(k) for k, v in new_record.items()]) +
                SQL_FROM + "DUAL"
            ) + " a" +
            SQL_LEFT_JOIN + sql_iso(
                SQL_SELECT + "'dummy' exist " +
                SQL_FROM + quote_column(table_name) +
                SQL_WHERE + condition +
                SQL_LIMIT + SQL_ONE
            ) + " b ON " + SQL_TRUE + SQL_WHERE + " exist " + SQL_IS_NULL
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
                "INSERT INTO " + quote_column(table_name) +
                sql_iso(sql_list([quote_column(k) for k in keys])) +
                " VALUES " + sql_list([
                sql_iso(sql_list([quote_value(r[k]) for k in keys]))
                for r in records
            ])
            )
            self.execute(command)
        except Exception as e:
            Log.error("problem with record: {{record}}", record=records, cause=e)

    def update(self, table_name, where_slice, new_values):
        """
        where_slice - A Data WHICH WILL BE USED TO MATCH ALL IN table
                      eg {"id": 42}
        new_values  - A dict WITH COLUMN NAME, COLUMN VALUE PAIRS TO SET
        """
        new_values = quote_param(new_values)

        where_clause = SQL_AND.join([
            quote_column(k) + "=" + quote_value(v) if v != None else quote_column(k) + SQL_IS_NULL
            for k, v in where_slice.items()
        ])

        command = (
            "UPDATE " + quote_column(table_name) + "\n" +
            "SET " +
            sql_list([quote_column(k) + "=" + v for k, v in new_values.items()]) +
            SQL_WHERE +
            where_clause
        )
        self.execute(command, {})

    def sort2sqlorderby(self, sort):
        sort = jx.normalize_sort_parameters(sort)
        return sql_list([quote_column(s.field) + (SQL_DESC if s.sort == -1 else SQL_ASC) for s in sort])

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
            sql = expand_template(sql, quote_param(param))

    # We have no way to execute an entire SQL file in bulk, so we
    # have to shell out to the commandline client.
    args = [
        "mysql",
        "-h{0}".format(host),
        "-u{0}".format(username),
        "-p{0}".format(password)
    ]
    if schema:
        args.append("{0}".format(schema))

    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=-1
        )
        if is_text(sql):
            sql = sql.encode("utf8")
        (output, _) = proc.communicate(sql)
    except Exception as e:
        raise Log.error("Can not call \"mysql\"", e)

    if proc.returncode:
        if len(sql) > 10000:
            sql = "<" + text_type(len(sql)) + " bytes of sql>"
        Log.error(
            "Unable to execute sql: return code {{return_code}}, {{output}}:\n {{sql}}\n",
            sql=indent(sql),
            return_code=proc.returncode,
            output=output
        )

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
    file = File(filename)
    if file.extension == 'zip':
        sql = file.read_zipfile()
    else:
        sql = File(filename).read()

    if ignore_errors:
        with suppress_exception:
            execute_sql(sql=sql, kwargs=kwargs)
    else:
        execute_sql(sql=sql, kwargs=kwargs)

ESCAPE_DCT = {
    u"\\": u"\\\\",
    u"\0": u"\\0",
    u"\"": u'\\"',
    u"\'": u"''",
    u"\b": u"\\b",
    u"\f": u"\\f",
    u"\n": u"\\n",
    u"\r": u"\\r",
    u"\t": u"\\t"
}


def quote_value(value):
    """
    convert values to mysql code for the same
    mostly delegate directly to the mysql lib, but some exceptions exist
    """
    try:
        if value == None:
            return SQL_NULL
        elif isinstance(value, SQL):
            return quote_sql(value.template, value.param)
        elif is_text(value):
            return SQL("'" + "".join(ESCAPE_DCT.get(c, c) for c in value) + "'")
        elif is_data(value):
            return quote_value(json_encode(value))
        elif is_number(value):
            return SQL(text_type(value))
        elif isinstance(value, datetime):
            return SQL("str_to_date('" + value.strftime("%Y%m%d%H%M%S.%f") + "', '%Y%m%d%H%i%s.%f')")
        elif isinstance(value, Date):
            return SQL("str_to_date('" + value.format("%Y%m%d%H%M%S.%f") + "', '%Y%m%d%H%i%s.%f')")
        elif hasattr(value, '__iter__'):
            return quote_value(json_encode(value))
        else:
            return quote_value(text_type(value))
    except Exception as e:
        Log.error("problem quoting SQL {{value}}", value=repr(value), cause=e)


def quote_column(column_name, table=None):
    if column_name == None:
        Log.error("missing column_name")
    elif is_text(column_name):
        if table:
            return join_column(table, column_name)
        else:
            return SQL("`" + '`.`'.join(split_field(column_name)) + "`")  # MYSQL QUOTE OF COLUMN NAMES
    elif is_binary(column_name):
        return quote_column(column_name.decode('utf8'), table)
    elif is_list(column_name):
        if table:
            return sql_list(join_column(table, c) for c in column_name)
        return sql_list(quote_column(c) for c in column_name)
    else:
        # ASSUME {"name":name, "value":value} FORM
        return SQL(sql_alias(column_name.value, quote_column(column_name.name)))


def quote_sql(value, param=None):
    """
    USED TO EXPAND THE PARAMETERS TO THE SQL() OBJECT
    """
    try:
        if isinstance(value, SQL):
            if not param:
                return value
            param = {k: quote_sql(v) for k, v in param.items()}
            return SQL(expand_template(value, param))
        elif is_text(value):
            return SQL(value)
        elif is_data(value):
            return quote_value(json_encode(value))
        elif hasattr(value, '__iter__'):
            return quote_list(value)
        else:
            return text_type(value)
    except Exception as e:
        Log.error("problem quoting SQL", e)


def quote_param(param):
    return {k: quote_value(v) for k, v in param.items()}


def quote_list(values):
    return sql_iso(sql_list(map(quote_value, values)))



def utf8_to_unicode(v):
    try:
        if is_binary(v):
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
        return {"terms": {term: values}}


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

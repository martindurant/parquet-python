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

import os
import re
import sqlite3
import sys
from collections import Mapping

from mo_future import text_type
from mo_dots import Data, coalesce
from mo_files import File
from mo_logs import Log
from mo_logs.exceptions import Except, extract_stack, ERROR
from mo_logs.strings import quote
from mo_math.stats import percentile
from mo_threads import Queue, Signal, Thread
from mo_times import Date, Duration
from mo_times.timer import Timer

from pyLibrary import convert
from pyLibrary.sql import DB, SQL

DEBUG = False
DEBUG_INSERT = False

_load_extension_warning_sent = False
_upgraded = False


def _upgrade():
    global _upgraded
    _upgraded = True
    try:
        import sys

        sqlite_dll = File.new_instance(sys.exec_prefix, "dlls/sqlite3.dll")
        python_dll = File("pyLibrary/vendor/sqlite/sqlite3.dll")
        if python_dll.read_bytes() != sqlite_dll.read_bytes():
            backup = sqlite_dll.backup()
            File.copy(python_dll, sqlite_dll)
    except Exception as e:
        Log.warning("could not upgrade python's sqlite", cause=e)


class Sqlite(DB):
    """
    Allows multi-threaded access
    Loads extension functions (like SQRT)
    """

    canonical = None

    def __init__(self, filename=None, db=None, upgrade=True):
        """
        :param db:  Optional, wrap a sqlite db in a thread
        :return: Multithread-safe database
        """
        if upgrade and not _upgraded:
            _upgrade()

        self.filename = filename
        self.db = db
        self.queue = Queue("sql commands")   # HOLD (command, result, signal) PAIRS
        self.worker = Thread.run("sqlite db thread", self._worker)
        self.get_trace = DEBUG
        self.upgrade = upgrade

    def _enhancements(self):
        def regex(pattern, value):
            return 1 if re.match(pattern+"$", value) else 0
        con = self.db.create_function("regex", 2, regex)

        class Percentile(object):
            def __init__(self, percentile):
                self.percentile=percentile
                self.acc=[]

            def step(self, value):
                self.acc.append(value)

            def finalize(self):
                return percentile(self.acc, self.percentile)

        con.create_aggregate("percentile", 2, Percentile)

    def execute(self, command):
        """
        COMMANDS WILL BE EXECUTED IN THE ORDER THEY ARE GIVEN
        BUT CAN INTERLEAVE WITH OTHER TREAD COMMANDS
        :param command: COMMAND FOR SQLITE
        :return: None
        """
        if DEBUG:  # EXECUTE IMMEDIATELY FOR BETTER STACK TRACE
            return self.query(command)

        if self.get_trace:
            trace = extract_stack(1)
        else:
            trace = None
        self.queue.add((command, None, None, trace))

    def query(self, command):
        """
        WILL BLOCK CALLING THREAD UNTIL THE command IS COMPLETED
        :param command: COMMAND FOR SQLITE
        :return: list OF RESULTS
        """
        if not self.worker:
            self.worker = Thread.run("sqlite db thread", self._worker)

        signal = Signal()
        result = Data()
        self.queue.add((command, result, signal, None))
        signal.wait()
        if result.exception:
            Log.error("Problem with Sqlite call", cause=result.exception)
        return result

    def _worker(self, please_stop):
        global _load_extension_warning_sent

        if DEBUG:
            Log.note("Sqlite version {{version}}", version=sqlite3.sqlite_version)
        if Sqlite.canonical:
            self.db = Sqlite.canonical
        else:
            self.db = sqlite3.connect(coalesce(self.filename, ':memory:'), check_same_thread = False)

            library_loc = File.new_instance(sys.modules[__name__].__file__, "../..")
            full_path = File.new_instance(library_loc, "vendor/sqlite/libsqlitefunctions.so").abspath
            try:
                trace = extract_stack(0)[0]
                if self.upgrade:
                    if os.name == 'nt':
                        file = File.new_instance(trace["file"], "../../vendor/sqlite/libsqlitefunctions.so")
                    else:
                        file = File.new_instance(trace["file"], "../../vendor/sqlite/libsqlitefunctions")

                    full_path = file.abspath
                    self.db.enable_load_extension(True)
                    self.db.execute("SELECT load_extension(" + self.quote_value(full_path) + ")")
            except Exception as e:
                if not _load_extension_warning_sent:
                    _load_extension_warning_sent = True
                    Log.warning("Could not load {{file}}}, doing without. (no SQRT for you!)", file=full_path, cause=e)

        try:
            while not please_stop:
                command, result, signal, trace = self.queue.pop(till=please_stop)

                if DEBUG_INSERT and command.strip().lower().startswith("insert"):
                    Log.note("Running command\n{{command|indent}}", command=command)
                if DEBUG and not command.strip().lower().startswith("insert"):
                    Log.note("Running command\n{{command|indent}}", command=command)
                with Timer("Run command", debug=DEBUG):
                    if signal is not None:
                        try:
                            curr = self.db.execute(command)
                            self.db.commit()
                            result.meta.format = "table"
                            result.header = [d[0] for d in curr.description] if curr.description else None
                            result.data = curr.fetchall()
                            if DEBUG and result.data:
                                text = convert.table2csv(list(result.data))
                                Log.note("Result:\n{{data}}", data=text)
                        except Exception as e:
                            e = Except.wrap(e)
                            result.exception = Except(ERROR, "Problem with\n{{command|indent}}", command=command, cause=e)
                        finally:
                            signal.go()
                    else:
                        try:
                            self.db.execute(command)
                            self.db.commit()
                        except Exception as e:
                            e = Except.wrap(e)
                            e.cause = Except(
                                type=ERROR,
                                template="Bad call to Sqlite",
                                trace=trace
                            )
                            Log.warning("Failure to execute", cause=e)

        except Exception as e:
            if not please_stop:
                Log.error("Problem with sql thread", e)
        finally:
            if DEBUG:
                Log.note("Database is closed")
            self.db.commit()
            self.db.close()

    def quote_column(self, column_name, table=None):
        return quote_column(column_name, table)

    def quote_value(self, value):
        return quote_value(value)

    def create_new_functions(self):

        def regexp(pattern, item):
            reg = re.compile(pattern)
            return reg.search(item) is not None

        self.db.create_function("REGEXP", 2, regexp)

_no_need_to_quote = re.compile(r"^\w+$", re.UNICODE)


def quote_column(column_name, table=None):
    if not isinstance(column_name, text_type):
        Log.error("expecting a name")
    if table != None:
        return SQL(quote(table) + "." + quote(column_name))
    else:
        if _no_need_to_quote.match(column_name):
            return SQL(column_name)
        return SQL(quote(column_name))


def quote_table(column):
    if _no_need_to_quote.match(column):
        return column
    return quote(column)


def quote_value(value):
    if isinstance(value, (Mapping, list)):
        return "."
    elif isinstance(value, Date):
        return text_type(value.unix)
    elif isinstance(value, Duration):
        return text_type(value.seconds)
    elif isinstance(value, text_type):
        return "'" + value.replace("'", "''") + "'"
    elif value == None:
        return "NULL"
    elif value is True:
        return "1"
    elif value is False:
        return "0"
    else:
        return text_type(value)

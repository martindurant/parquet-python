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

import sys
from collections import Mapping

from mo_dots import Data, listwrap, unwraplist, set_default, Null
from mo_future import text_type, PY3
from mo_logs.strings import indent, expand_template


FATAL = "FATAL"
ERROR = "ERROR"
WARNING = "WARNING"
ALARM = "ALARM"
UNEXPECTED = "UNEXPECTED"
NOTE = "NOTE"


class Except(Exception):

    @staticmethod
    def new_instance(desc):
        return Except(
            desc.type,
            desc.template,
            desc.params,
            [Except.new_instance(c) for c in listwrap(desc.cause)],
            desc.trace
        )

    def __init__(self, type=ERROR, template=Null, params=Null, cause=Null, trace=Null, **kwargs):
        Exception.__init__(self)
        self.type = type
        self.template = template
        self.params = set_default(kwargs, params)
        self.cause = cause

        if not trace:
            self.trace=extract_stack(2)
        else:
            self.trace = trace

    @classmethod
    def wrap(cls, e, stack_depth=0):
        if e == None:
            return Null
        elif isinstance(e, (list, Except)):
            return e
        elif isinstance(e, Mapping):
            e.cause = unwraplist([Except.wrap(c) for c in listwrap(e.cause)])
            return Except(**e)
        else:
            if hasattr(e, "message") and e.message:
                cause = Except(ERROR, text_type(e.message), trace=_extract_traceback(0))
            else:
                cause = Except(ERROR, text_type(e), trace=_extract_traceback(0))

            trace = extract_stack(stack_depth + 2)  # +2 = to remove the caller, and it's call to this' Except.wrap()
            cause.trace.extend(trace)
            return cause

    @property
    def message(self):
        return expand_template(self.template, self.params)

    def __contains__(self, value):
        if isinstance(value, text_type):
            if self.template.find(value) >= 0 or self.message.find(value) >= 0:
                return True

        if self.type == value:
            return True
        for c in listwrap(self.cause):
            if value in c:
                return True
        return False

    def __unicode__(self):
        output = self.type + ": " + self.template + "\n"
        if self.params:
            output = expand_template(output, self.params)

        if self.trace:
            output += indent(format_trace(self.trace))

        if self.cause:
            cause_strings = []
            for c in listwrap(self.cause):
                with suppress_exception:
                    cause_strings.append(text_type(c))


            output += "caused by\n\t" + "and caused by\n\t".join(cause_strings)

        return output

    if PY3:
        def __str__(self):
            return self.__unicode__()
    else:
        def __str__(self):
            return self.__unicode__().encode('latin1', 'replace')

    def __data__(self):
        return Data(
            type=self.type,
            template=self.template,
            params=self.params,
            cause=self.cause,
            trace=self.trace
        )


def extract_stack(start=0):
    """
    SNAGGED FROM traceback.py
    Altered to return Data

    Extract the raw traceback from the current stack frame.

    Each item in the returned list is a quadruple (filename,
    line number, function name, text), and the entries are in order
    from newest to oldest
    """
    try:
        raise ZeroDivisionError
    except ZeroDivisionError:
        trace = sys.exc_info()[2]
        f = trace.tb_frame.f_back

    for i in range(start):
        f = f.f_back

    stack = []
    n = 0
    while f is not None:
        stack.append({
            "depth": n,
            "line": f.f_lineno,
            "file": f.f_code.co_filename,
            "method": f.f_code.co_name
        })
        f = f.f_back
        n += 1
    return stack


def _extract_traceback(start):
    """
    SNAGGED FROM traceback.py

    RETURN list OF dicts DESCRIBING THE STACK TRACE
    """
    tb = sys.exc_info()[2]
    for i in range(start):
        tb = tb.tb_next

    trace = []
    n = 0
    while tb is not None:
        f = tb.tb_frame
        trace.append({
            "depth": n,
            "file": f.f_code.co_filename,
            "line": tb.tb_lineno,
            "method": f.f_code.co_name
        })
        tb = tb.tb_next
        n += 1
    trace.reverse()
    return trace


def format_trace(tbs, start=0):
    trace = []
    for d in tbs[start::]:
        item = expand_template('File "{{file}}", line {{line}}, in {{method}}\n', d)
        trace.append(item)
    return "".join(trace)


class Suppress(object):
    """
    IGNORE EXCEPTIONS
    """

    def __init__(self, exception_type):
        self.type = exception_type

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val or isinstance(exc_val, self.type):
            return True

suppress_exception = Suppress(Exception)


class Explanation(object):
    """
    EXPLAIN THE ACTION BEING TAKEN
    IF THERE IS AN EXCEPTION WRAP IT WITH THE EXPLANATION
    CHAIN EXCEPTION AND RE-RAISE
    """

    def __init__(
        self,
        template,  # human readable template
        debug=False,
        **more_params
    ):
        self.debug = debug
        self.template = template
        self.more_params = more_params

    def __enter__(self):
        if self.debug:
            from mo_logs import Log
            Log.note(self.template, default_params=self.more_params, stack_depth=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, Exception):
            from mo_logs import Log

            Log.error(
                template="Failure in " + self.template,
                default_params=self.more_params,
                cause=exc_val,
                stack_depth=1
            )

            return True


class WarnOnException(object):
    """
    EXPLAIN THE ACTION BEING TAKEN
    IF THERE IS AN EXCEPTION WRAP ISSUE A WARNING
    """

    def __init__(
        self,
        template,  # human readable template
        debug=False,
        **more_params
    ):
        self.debug = debug
        self.template = template
        self.more_params = more_params

    def __enter__(self):
        if self.debug:
            from mo_logs import Log
            Log.note(self.template, default_params=self.more_params, stack_depth=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, Exception):
            from mo_logs import Log

            Log.warning(
                template="Ignored failure while " + self.template,
                default_params=self.more_params,
                cause=exc_val,
                stack_depth=1
            )

            return True


class AssertNoException(object):
    """
    EXPECT NO EXCEPTION IN THIS BLOCK
    """

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, Exception):
            from mo_logs import Log

            Log.error(
                template="Not expected to fail",
                cause=exc_val,
                stack_depth=1
            )

            return True

assert_no_exception = AssertNoException()

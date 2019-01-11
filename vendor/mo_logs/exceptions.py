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

from mo_future import is_text, is_binary
import sys

from mo_dots import Data, Null, is_data, listwrap, unwraplist
from mo_future import PY3, text_type
from mo_logs.strings import CR, expand_template, indent

FATAL = "FATAL"
ERROR = "ERROR"
WARNING = "WARNING"
ALARM = "ALARM"
UNEXPECTED = "UNEXPECTED"
NOTE = "NOTE"


class LogItem(object):

    def __init__(self, context, format, template, params):
        self.context = context
        self.format = format
        self.template = template
        self.params = params

    def __data__(self):
        return Data(self.__dict__)


class Except(Exception, LogItem):

    @staticmethod
    def new_instance(desc):
        return Except(
            context=desc.context,
            template=desc.template,
            params=desc.params,
            cause=[Except.new_instance(c) for c in listwrap(desc.cause)],
            trace=desc.trace
        )

    def __init__(self, context=ERROR, template=Null, params=Null, cause=Null, trace=Null, **_):
        if context == None:
            raise ValueError("expecting context to not be None")

        self.cause = Except.wrap(cause)

        Exception.__init__(self)
        LogItem.__init__(
            self,
            context=context,
            format=None,
            template=template,
            params=params
        )

        if not trace:
            self.trace = extract_stack(2)
        else:
            self.trace = trace

    @classmethod
    def wrap(cls, e, stack_depth=0):
        """
        ENSURE THE STACKTRACE AND CAUSAL CHAIN IS CAPTURED, PLUS ADD FEATURES OF Except

        :param e: AN EXCEPTION OF ANY TYPE
        :param stack_depth: HOW MANY CALLS TO TAKE OFF THE TOP OF THE STACK TRACE
        :return: A Except OBJECT OF THE SAME
        """
        if e == None:
            return Null
        elif isinstance(e, (list, Except)):
            return e
        elif is_data(e):
            e.cause = unwraplist([Except.wrap(c) for c in listwrap(e.cause)])
            return Except(**e)
        else:
            tb = getattr(e, '__traceback__', None)
            if tb is not None:
                trace = _parse_traceback(tb)
            else:
                trace = _extract_traceback(0)

            cause = Except.wrap(getattr(e, '__cause__', None))
            if hasattr(e, "message") and e.message:
                output = Except(context=ERROR, template=text_type(e.message), trace=trace, cause=cause)
            else:
                output = Except(context=ERROR, template=text_type(e), trace=trace, cause=cause)

            trace = extract_stack(stack_depth + 2)  # +2 = to remove the caller, and it's call to this' Except.wrap()
            output.trace.extend(trace)
            return output

    @property
    def message(self):
        return expand_template(self.template, self.params)

    def __contains__(self, value):
        if is_text(value):
            if self.template.find(value) >= 0 or self.message.find(value) >= 0:
                return True

        if self.context == value:
            return True
        for c in listwrap(self.cause):
            if value in c:
                return True
        return False

    def __unicode__(self):
        output = self.context + ": " + self.template + CR
        if self.params:
            output = expand_template(output, self.params)

        if self.trace:
            output += indent(format_trace(self.trace))

        if self.cause:
            cause_strings = []
            for c in listwrap(self.cause):
                try:
                    cause_strings.append(text_type(c))
                except Exception as e:
                    sys.stderr("Problem serializing cause"+text_type(c))

            output += "caused by\n\t" + "and caused by\n\t".join(cause_strings)

        return output

    if PY3:
        def __str__(self):
            return self.__unicode__()
    else:
        def __str__(self):
            return self.__unicode__().encode('latin1', 'replace')

    def __data__(self):
        output = Data({k:getattr(self,k) for k in vars(self)})
        output.cause=unwraplist([c.__data__() for c in listwrap(output.cause)])
        return output


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
    while f is not None:
        stack.append({
            "line": f.f_lineno,
            "file": f.f_code.co_filename,
            "method": f.f_code.co_name
        })
        f = f.f_back
    return stack


def _extract_traceback(start):
    """
    SNAGGED FROM traceback.py

    RETURN list OF dicts DESCRIBING THE STACK TRACE
    """
    tb = sys.exc_info()[2]
    for i in range(start):
        tb = tb.tb_next
    return _parse_traceback(tb)


def _parse_traceback(tb):
    trace = []
    while tb is not None:
        f = tb.tb_frame
        trace.append({
            "file": f.f_code.co_filename,
            "line": tb.tb_lineno,
            "method": f.f_code.co_name
        })
        tb = tb.tb_next
    trace.reverse()
    return trace


def format_trace(tbs, start=0):
    return "".join(
        expand_template('File "{{file}}", line {{line}}, in {{method}}\n', d)
        for d in tbs[start::]
    )


class Suppress(object):
    """
    IGNORE EXCEPTIONS
    """

    def __init__(self, exception_type):
        self.context = exception_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val or isinstance(exc_val, self.context):
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

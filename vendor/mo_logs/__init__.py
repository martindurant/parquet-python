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
import platform
from collections import Mapping
from datetime import datetime

import sys

from mo_dots import coalesce, listwrap, wrap, unwrap, unwraplist, set_default, FlatList
from mo_future import text_type, PY3
from mo_logs import constants
from mo_logs.exceptions import Except, suppress_exception
from mo_logs.strings import indent

_Thread = None

class Log(object):
    """
    FOR STRUCTURED LOGGING AND EXCEPTION CHAINING
    """
    trace = False
    main_log = None
    logging_multi = None
    profiler = None   # simple pypy-friendly profiler
    cprofiler = None  # screws up with pypy, but better than nothing
    cprofiler_stats = None
    error_mode = False  # prevent error loops

    @classmethod
    def start(cls, settings=None):
        """
        RUN ME FIRST TO SETUP THE THREADED LOGGING
        http://victorlin.me/2012/08/good-logging-practice-in-python/

        log       - LIST OF PARAMETERS FOR LOGGER(S)
        trace     - SHOW MORE DETAILS IN EVERY LOG LINE (default False)
        cprofile  - True==ENABLE THE C-PROFILER THAT COMES WITH PYTHON (default False)
                    USE THE LONG FORM TO SET THE FILENAME {"enabled": True, "filename": "cprofile.tab"}
        profile   - True==ENABLE pyLibrary SIMPLE PROFILING (default False) (eg with Profiler("some description"):)
                    USE THE LONG FORM TO SET FILENAME {"enabled": True, "filename": "profile.tab"}
        constants - UPDATE MODULE CONSTANTS AT STARTUP (PRIMARILY INTENDED TO CHANGE DEBUG STATE)
        """
        global _Thread

        if not settings:
            return
        settings = wrap(settings)

        Log.stop()

        cls.settings = settings
        cls.trace = coalesce(settings.trace, False)
        if cls.trace:
            from mo_threads import Thread as _Thread
            _ = _Thread

        if settings.cprofile is False:
            settings.cprofile = {"enabled": False}
        elif settings.cprofile is True or (isinstance(settings.cprofile, Mapping) and settings.cprofile.enabled):
            if isinstance(settings.cprofile, bool):
                settings.cprofile = {"enabled": True, "filename": "cprofile.tab"}

            import cProfile

            cls.cprofiler = cProfile.Profile()
            cls.cprofiler.enable()

        if settings.profile is True or (isinstance(settings.profile, Mapping) and settings.profile.enabled):
            from mo_logs import profiles

            if isinstance(settings.profile, bool):
                profiles.ON = True
                settings.profile = {"enabled": True, "filename": "profile.tab"}

            if settings.profile.enabled:
                profiles.ON = True

        if settings.constants:
            constants.set(settings.constants)

        if settings.log:
            cls.logging_multi = StructuredLogger_usingMulti()
            from mo_logs.log_usingThread import StructuredLogger_usingThread
            cls.main_log = StructuredLogger_usingThread(cls.logging_multi)

            for log in listwrap(settings.log):
                Log.add_log(Log.new_instance(log))

        if settings.cprofile.enabled==True:
            Log.alert("cprofiling is enabled, writing to {{filename}}", filename=os.path.abspath(settings.cprofile.filename))

    @classmethod
    def stop(cls):
        from mo_logs import profiles

        if cls.cprofiler and hasattr(cls, "settings"):
            if cls.cprofiler == None:
                from mo_threads import Queue

                cls.cprofiler_stats = Queue("cprofiler stats")  # ACCUMULATION OF STATS FROM ALL THREADS

            import pstats
            cls.cprofiler_stats.add(pstats.Stats(cls.cprofiler))
            write_profile(cls.settings.cprofile, cls.cprofiler_stats.pop_all())

        if profiles.ON and hasattr(cls, "settings"):
            profiles.write(cls.settings.profile)
        cls.main_log.stop()
        cls.main_log = StructuredLogger_usingStream(sys.stdout)

    @classmethod
    def new_instance(cls, settings):
        settings = wrap(settings)

        if settings["class"]:
            if settings["class"].startswith("logging.handlers."):
                from mo_logs.log_usingLogger import StructuredLogger_usingLogger

                return StructuredLogger_usingLogger(settings)
            else:
                with suppress_exception:
                    from mo_logs.log_usingLogger import make_log_from_settings

                    return make_log_from_settings(settings)
                  # OH WELL :(

        if settings.log_type == "file" or settings.file:
            return StructuredLogger_usingFile(settings.file)
        if settings.log_type == "file" or settings.filename:
            return StructuredLogger_usingFile(settings.filename)
        if settings.log_type == "console":
            from mo_logs.log_usingThreadedStream import StructuredLogger_usingThreadedStream
            return StructuredLogger_usingThreadedStream(sys.stdout)
        if settings.log_type == "stream" or settings.stream:
            from mo_logs.log_usingThreadedStream import StructuredLogger_usingThreadedStream
            return StructuredLogger_usingThreadedStream(settings.stream)
        if settings.log_type == "elasticsearch" or settings.stream:
            from mo_logs.log_usingElasticSearch import StructuredLogger_usingElasticSearch
            return StructuredLogger_usingElasticSearch(settings)
        if settings.log_type == "email":
            from mo_logs.log_usingEmail import StructuredLogger_usingEmail
            return StructuredLogger_usingEmail(settings)
        if settings.log_type == "ses":
            from mo_logs.log_usingSES import StructuredLogger_usingSES
            return StructuredLogger_usingSES(settings)
        if settings.log_type.lower() in ["nothing", "none", "null"]:
            from mo_logs.log_usingNothing import StructuredLogger
            return StructuredLogger()

        Log.error("Log type of {{log_type|quote}} is not recognized", log_type=settings.log_type)

    @classmethod
    def add_log(cls, log):
        cls.logging_multi.add_log(log)

    @classmethod
    def note(
        cls,
        template,
        default_params={},
        stack_depth=0,
        log_context=None,
        **more_params
    ):
        """
        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        if not isinstance(template, text_type):
            Log.error("Log.note was expecting a unicode template")

        if len(template) > 10000:
            template = template[:10000]

        params = dict(unwrap(default_params), **more_params)

        log_params = set_default({
            "template": template,
            "params": params,
            "timestamp": datetime.utcnow(),
            "machine": machine_metadata
        }, log_context, {"context": exceptions.NOTE})

        if not template.startswith("\n") and template.find("\n") > -1:
            template = "\n" + template

        if cls.trace:
            log_template = "{{machine.name}} (pid {{machine.pid}}) - {{timestamp|datetime}} - {{thread.name}} - \"{{location.file}}:{{location.line}}\" ({{location.method}}) - " + template.replace("{{", "{{params.")
            f = sys._getframe(stack_depth + 1)
            log_params.location = {
                "line": f.f_lineno,
                "file": text_type(f.f_code.co_filename.split(os.sep)[-1]),
                "method": text_type(f.f_code.co_name)
            }
            thread = _Thread.current()
            log_params.thread = {"name": thread.name, "id": thread.id}
        else:
            log_template = "{{timestamp|datetime}} - " + template.replace("{{", "{{params.")

        cls.main_log.write(log_template, log_params)

    @classmethod
    def unexpected(
        cls,
        template,
        default_params={},
        cause=None,
        stack_depth=0,
        log_context=None,
        **more_params
    ):
        """
        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param cause: *Exception* for chaining
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        if isinstance(default_params, BaseException):
            cause = default_params
            default_params = {}

        params = dict(unwrap(default_params), **more_params)

        if cause and not isinstance(cause, Except):
            cause = Except(exceptions.UNEXPECTED, text_type(cause), trace=exceptions._extract_traceback(0))

        trace = exceptions.extract_stack(1)
        e = Except(exceptions.UNEXPECTED, template, params, cause, trace)
        Log.note(
            "{{error}}",
            error=e,
            log_context=set_default({"context": exceptions.WARNING}, log_context),
            stack_depth=stack_depth + 1
        )

    @classmethod
    def alarm(
        cls,
        template,
        default_params={},
        stack_depth=0,
        log_context=None,
        **more_params
    ):
        """
        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        # USE replace() AS POOR MAN'S CHILD TEMPLATE

        template = ("*" * 80) + "\n" + indent(template, prefix="** ").strip() + "\n" + ("*" * 80)
        Log.note(
            template,
            default_params=default_params,
            stack_depth=stack_depth + 1,
            log_context=set_default({"context": exceptions.ALARM}, log_context),
            **more_params
        )

    @classmethod
    def alert(
        cls,
        template,
        default_params={},
        stack_depth=0,
        log_context=None,
        **more_params
    ):
        """
        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        return Log.alarm(
            template,
            default_params=default_params,
            stack_depth=stack_depth + 1,
            log_context=set_default({"context": exceptions.ALARM}, log_context),
            **more_params
        )

    @classmethod
    def warning(
        cls,
        template,
        default_params={},
        cause=None,
        stack_depth=0,
        log_context=None,
        **more_params
    ):
        """
        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param cause: *Exception* for chaining
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        if not isinstance(template, text_type):
            Log.error("Log.note was expecting a unicode template")

        if isinstance(default_params, BaseException):
            cause = default_params
            default_params = {}

        if "values" in more_params.keys():
            Log.error("Can not handle a logging parameter by name `values`")
        params = dict(unwrap(default_params), **more_params)
        cause = unwraplist([Except.wrap(c) for c in listwrap(cause)])
        trace = exceptions.extract_stack(stack_depth + 1)

        e = Except(exceptions.WARNING, template, params, cause, trace)
        Log.note(
            "{{error|unicode}}",
            error=e,
            log_context=set_default({"context": exceptions.WARNING}, log_context),
            stack_depth=stack_depth + 1
        )


    @classmethod
    def error(
        cls,
        template,  # human readable template
        default_params={},  # parameters for template
        cause=None,  # pausible cause
        stack_depth=0,
        **more_params
    ):
        """
        raise an exception with a trace for the cause too

        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param cause: *Exception* for chaining
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        if not isinstance(template, text_type):
            sys.stderr.write(str("Log.error was expecting a unicode template"))
            Log.error("Log.error was expecting a unicode template")

        if default_params and isinstance(listwrap(default_params)[0], BaseException):
            cause = default_params
            default_params = {}

        params = dict(unwrap(default_params), **more_params)

        add_to_trace = False
        if cause == None:
            pass
        elif isinstance(cause, list):
            cause = []
            for c in listwrap(cause):  # CAN NOT USE LIST-COMPREHENSION IN PYTHON3 (EXTRA STACK DEPTH FROM THE IN-LINED GENERATOR)
                cause.append(Except.wrap(c, stack_depth=1))
            cause = FlatList(cause)
        elif isinstance(cause, BaseException):
            cause = Except.wrap(cause, stack_depth=1)
        else:
            Log.error("can only accept Exception , or list of exceptions")

        trace = exceptions.extract_stack(stack_depth + 1)

        if add_to_trace:
            cause[0].trace.extend(trace[1:])

        e = Except(exceptions.ERROR, template, params, cause, trace)
        raise_from_none(e)

    @classmethod
    def fatal(
        cls,
        template,  # human readable template
        default_params={},  # parameters for template
        cause=None,  # pausible cause
        stack_depth=0,
        log_context=None,
        **more_params
    ):
        """
        SEND TO STDERR

        :param template: *string* human readable string with placeholders for parameters
        :param default_params: *dict* parameters to fill in template
        :param cause: *Exception* for chaining
        :param stack_depth:  *int* how many calls you want popped off the stack to report the *true* caller
        :param log_context: *dict* extra key:value pairs for your convenience
        :param more_params: *any more parameters (which will overwrite default_params)
        :return:
        """
        if default_params and isinstance(listwrap(default_params)[0], BaseException):
            cause = default_params
            default_params = {}

        params = dict(unwrap(default_params), **more_params)

        cause = unwraplist([Except.wrap(c) for c in listwrap(cause)])
        trace = exceptions.extract_stack(stack_depth + 1)

        e = Except(exceptions.ERROR, template, params, cause, trace)
        str_e = text_type(e)

        error_mode = cls.error_mode
        with suppress_exception:
            if not error_mode:
                cls.error_mode = True
                Log.note(
                    "{{error|unicode}}",
                    error=e,
                    log_context=set_default({"context": exceptions.FATAL}, log_context),
                    stack_depth=stack_depth + 1
                )
        cls.error_mode = error_mode

        sys.stderr.write(str_e.encode('utf8'))


    def write(self):
        raise NotImplementedError


def write_profile(profile_settings, stats):
    from pyLibrary import convert
    from mo_files import File

    acc = stats[0]
    for s in stats[1:]:
        acc.add(s)

    stats = [{
        "num_calls": d[1],
        "self_time": d[2],
        "total_time": d[3],
        "self_time_per_call": d[2] / d[1],
        "total_time_per_call": d[3] / d[1],
        "file": (f[0] if f[0] != "~" else "").replace("\\", "/"),
        "line": f[1],
        "method": f[2].lstrip("<").rstrip(">")
    }
        for f, d, in acc.stats.iteritems()
    ]
    stats_file = File(profile_settings.filename, suffix=convert.datetime2string(datetime.now(), "_%Y%m%d_%H%M%S"))
    stats_file.write(convert.list2tab(stats))


# GET THE MACHINE METADATA
machine_metadata = wrap({
    "pid":  os.getpid(),
    "python": text_type(platform.python_implementation()),
    "os": text_type(platform.system() + platform.release()).strip(),
    "name": text_type(platform.node())
})


def raise_from_none(e):
    raise e

if PY3:
    exec("def raise_from_none(e):\n    raise e from None\n", globals(), locals())


from mo_logs.log_usingFile import StructuredLogger_usingFile
from mo_logs.log_usingMulti import StructuredLogger_usingMulti
from mo_logs.log_usingStream import StructuredLogger_usingStream


if not Log.main_log:
    Log.main_log = StructuredLogger_usingStream(sys.stdout)


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
from collections import namedtuple
from types import FunctionType

from mo_dots import Null, _get_attr, set_default
from mo_future import get_function_arguments, get_function_name, text_type
import mo_json
from mo_logs import Log
from mo_logs.exceptions import Except
from mo_math.randoms import Random
from mo_threads import Lock
from mo_times.dates import Date
from mo_times.durations import DAY


def get_class(path):
    try:
        #ASSUME DIRECT FROM MODULE
        output = __import__(".".join(path[0:-1]), globals(), locals(), [path[-1]], 0)
        return _get_attr(output, path[-1:])
        # return output
    except Exception as e:
        from mo_logs import Log

        Log.error("Could not find module {{module|quote}}",  module= ".".join(path))


def new_instance(settings):
    """
    MAKE A PYTHON INSTANCE

    `settings` HAS ALL THE `kwargs`, PLUS `class` ATTRIBUTE TO INDICATE THE CLASS TO CREATE
    """
    settings = set_default({}, settings)
    if not settings["class"]:
        Log.error("Expecting 'class' attribute with fully qualified class name")

    # IMPORT MODULE FOR HANDLER
    path = settings["class"].split(".")
    class_name = path[-1]
    path = ".".join(path[:-1])
    constructor = None
    try:
        temp = __import__(path, globals(), locals(), [class_name], 0)
        constructor = object.__getattribute__(temp, class_name)
    except Exception as e:
        Log.error("Can not find class {{class}}", {"class": path}, cause=e)

    settings['class'] = None
    try:
        return constructor(kwargs=settings)  # MAYBE IT TAKES A KWARGS OBJECT
    except Exception as e:
        pass

    try:
        return constructor(**settings)
    except Exception as e:
        Log.error("Can not create instance of {{name}}", name=".".join(path), cause=e)


def get_function_by_name(full_name):
    """
    RETURN FUNCTION
    """

    # IMPORT MODULE FOR HANDLER
    path = full_name.split(".")
    function_name = path[-1]
    path = ".".join(path[:-1])
    constructor = None
    try:
        temp = __import__(path, globals(), locals(), [function_name], -1)
        output = object.__getattribute__(temp, function_name)
        return output
    except Exception as e:
        Log.error("Can not find function {{name}}",  name= full_name, cause=e)





class cache(object):

    """
    :param func: ASSUME FIRST PARAMETER OF `func` IS `self`
    :param duration: USE CACHE IF LAST CALL WAS LESS THAN duration AGO
    :param lock: True if you want multithreaded monitor (default False)
    :return:
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], FunctionType):
            func = args[0]
            return wrap_function(_SimpleCache(), func)
        else:
            return object.__new__(cls)

    def __init__(self, duration=DAY, lock=False):
        self.timeout = duration
        if lock:
            self.locker = Lock()
        else:
            self.locker = _FakeLock()

    def __call__(self, func):
        return wrap_function(self, func)


class _SimpleCache(object):

    def __init__(self):
        self.timeout = Null
        self.locker = _FakeLock()


def wrap_function(cache_store, func_):
    attr_name = "_cache_for_" + func_.__name__

    func_args = get_function_arguments(func_)
    if len(func_args) > 0 and func_args[0] == "self":
        using_self = True
        func = lambda self, *args: func_(self, *args)
    else:
        using_self = False
        func = lambda self, *args: func_(*args)

    def output(*args, **kwargs):
        if kwargs:
            Log.error("Sorry, caching only works with ordered parameter, not keyword arguments")

        with cache_store.locker:
            if using_self:
                self = args[0]
                args = args[1:]
            else:
                self = cache_store

            now = Date.now()
            try:
                _cache = getattr(self, attr_name)
            except Exception:
                _cache = {}
                setattr(self, attr_name, _cache)

            if Random.int(100) == 0:
                # REMOVE OLD CACHE
                _cache = {k: v for k, v in _cache.items() if v.timeout == None or v.timeout > now}
                setattr(self, attr_name, _cache)

            timeout, key, value, exception = _cache.get(args, (Null, Null, Null, Null))

        if now >= timeout:
            value = func(self, *args)
            with cache_store.locker:
                _cache[args] = CacheElement(now + cache_store.timeout, args, value, None)
            return value

        if value == None:
            if exception == None:
                try:
                    value = func(self, *args)
                    with cache_store.locker:
                        _cache[args] = CacheElement(now + cache_store.timeout, args, value, None)
                    return value
                except Exception as e:
                    e = Except.wrap(e)
                    with cache_store.locker:
                        _cache[args] = CacheElement(now + cache_store.timeout, args, None, e)
                    raise e
            else:
                raise exception
        else:
            return value

    return output


CacheElement = namedtuple("CacheElement", ("timeout", "key", "value", "exception"))


class _FakeLock():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def value2quote(value):
    # RETURN PRETTY PYTHON CODE FOR THE SAME
    if is_text(value):
        return mo_json.quote(value)
    else:
        return text_type(repr(value))


class extenstion_method(object):
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

    def __call__(self, func):
        if self.name is not None:
            setattr(self.value, self.name, func)
            return func
        else:
            setattr(self.value, func.__name__, func)
            return func


def extend(cls):
    """
    DECORATOR TO ADD METHODS TO CLASSES
    :param cls: THE CLASS TO ADD THE METHOD TO
    :return:
    """
    def extender(func):
        setattr(cls, get_function_name(func), func)
        return func
    return extender


class MemorySample(object):

    def __init__(self, description, debug=False, **parameters):
        self.debug = debug
        if debug:
            try:
                import os
                import psutil
                import gc

                self.description = description
                self.params = parameters
                self.start_memory = None
                self.process = psutil.Process(os.getpid())
            except Exception as e:
                Log.warning("problem in memory measure", cause=e)

    def __enter__(self):
        if self.debug:
            try:
                gc.collect()
                self.start_memory = self.process.memory_info().rss
            except Exception as e:
                Log.warning("problem in memory measure", cause=e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debug:
            try:
                gc.collect()
                end_memory = self.process.memory_info().rss
                net_memory = end_memory-self.start_memory
                if net_memory > 100 * 1000 * 1000:
                    Log.warning(
                        "MEMORY WARNING (additional {{net_memory|comma}}bytes): "+self.description,
                        default_params=self.params,
                        net_memory=net_memory
                    )

                    from pympler import summary
                    from pympler import muppy
                    sum1 = sorted(summary.summarize(muppy.get_objects()), key=lambda r: -r[2])[:30]
                    Log.warning("{{data}}", data=sum1)
                elif end_memory > 1000*1000*1000:
                    Log.warning(
                        "MEMORY WARNING (over {{end_memory|comma}}bytes): "+self.description,
                        default_params=self.params,
                        end_memory=end_memory
                    )

                    from pympler import summary
                    from pympler import muppy
                    sum1 = sorted(summary.summarize(muppy.get_objects()), key=lambda r: -r[2])[:30]
                    Log.warning("{{data}}", data=sum1)

            except Exception as e:
                Log.warning("problem in memory measure", cause=e)

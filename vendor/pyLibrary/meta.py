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

import mo_json
from mo_future import text_type, get_function_arguments

from mo_dots import set_default, wrap, _get_attr, Null, coalesce
from mo_json import value2json
from mo_logs import Log
from mo_threads import Lock
from pyLibrary import convert
from types import FunctionType

from jx_python.expressions import jx_expression
from mo_logs.exceptions import Except
from mo_logs.strings import expand_template
from mo_math.randoms import Random
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
        temp = __import__(path, globals(), locals(), [class_name], -1)
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

    def output(*args):
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
                _cache = {k: v for k, v in _cache.items() if v[0]==None or v[0] > now}
                setattr(self, attr_name, _cache)

            timeout, key, value, exception = _cache.get(args, (Null, Null, Null, Null))

        if now >= timeout:
            value = func(self, *args)
            with cache_store.locker:
                _cache[args] = (now + cache_store.timeout, args, value, None)
            return value

        if value == None:
            if exception == None:
                try:
                    value = func(self, *args)
                    with cache_store.locker:
                        _cache[args] = (now + cache_store.timeout, args, value, None)
                    return value
                except Exception as e:
                    e = Except.wrap(e)
                    with cache_store.locker:
                        _cache[args] = (now + cache_store.timeout, args, None, e)
                    raise e
            else:
                raise exception
        else:
            return value

    return output


class _FakeLock():


    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def DataClass(name, columns, constraint=True):
    """
    Use the DataClass to define a class, but with some extra features:
    1. restrict the datatype of property
    2. restrict if `required`, or if `nulls` are allowed
    3. generic constraints on object properties

    It is expected that this class become a real class (or be removed) in the
    long term because it is expensive to use and should only be good for
    verifying program correctness, not user input.

    :param name: Name of the class we are creating
    :param columns: Each columns[i] has properties {
            "name",     - (required) name of the property
            "required", - False if it must be defined (even if None)
            "nulls",    - True if property can be None, or missing
            "default",  - A default value, if none is provided
            "type"      - a Python datatype
        }
    :param constraint: a JSON query Expression for extra constraints
    :return: The class that has been created
    """

    columns = wrap([{"name": c, "required": True, "nulls": False, "type": object} if isinstance(c, text_type) else c for c in columns])
    slots = columns.name
    required = wrap(filter(lambda c: c.required and not c.nulls and not c.default, columns)).name
    nulls = wrap(filter(lambda c: c.nulls, columns)).name
    defaults = {c.name: coalesce(c.default, None) for c in columns}
    types = {c.name: coalesce(c.type, object) for c in columns}

    code = expand_template(
"""
from __future__ import unicode_literals
from collections import Mapping

meta = None
types_ = {{types}}
defaults_ = {{defaults}}

class {{class_name}}(Mapping):
    __slots__ = {{slots}}


    def _constraint(row, rownum, rows):
        return {{constraint_expr}}

    def __init__(self, **kwargs):
        if not kwargs:
            return

        for s in {{slots}}:
            object.__setattr__(self, s, kwargs.get(s, {{defaults}}.get(s, None)))

        missed = {{required}}-set(kwargs.keys())
        if missed:
            Log.error("Expecting properties {"+"{missed}}", missed=missed)

        illegal = set(kwargs.keys())-set({{slots}})
        if illegal:
            Log.error("{"+"{names}} are not a valid properties", names=illegal)

        if not self._constraint(0, [self]):
            Log.error("constraint not satisfied {"+"{expect}}\\n{"+"{value|indent}}", expect={{constraint}}, value=self)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)
        return self

    def __setattr__(self, item, value):
        if item not in {{slots}}:
            Log.error("{"+"{item|quote}} not valid attribute", item=item)
        object.__setattr__(self, item, value)
        if not self._constraint(0, [self]):
            Log.error("constraint not satisfied {"+"{expect}}\\n{"+"{value|indent}}", expect={{constraint}}, value=self)

    def __getattr__(self, item):
        Log.error("{"+"{item|quote}} not valid attribute", item=item)

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other):
        if isinstance(other, {{class_name}}) and dict(self)==dict(other) and self is not other:
            Log.error("expecting to be same object")
        return self is other

    def __dict__(self):
        return {k: getattr(self, k) for k in {{slots}}}

    def items(self):
        return ((k, getattr(self, k)) for k in {{slots}})

    def __copy__(self):
        _set = object.__setattr__
        output = object.__new__({{class_name}})
        {{assign}}
        return output

    def __iter__(self):
        return {{slots}}.__iter__()

    def __len__(self):
        return {{len_slots}}

    def __str__(self):
        return str({{dict}})

""",
        {
            "class_name": name,
            "slots": "(" + (", ".join(convert.value2quote(s) for s in slots)) + ")",
            "required": "{" + (", ".join(convert.value2quote(s) for s in required)) + "}",
            "nulls": "{" + (", ".join(convert.value2quote(s) for s in nulls)) + "}",
            "defaults": jx_expression({"literal": defaults}).to_python(),
            "len_slots": len(slots),
            "dict": "{" + (", ".join(convert.value2quote(s) + ": self." + s for s in slots)) + "}",
            "assign": "; ".join("_set(output, "+convert.value2quote(s)+", self."+s+")" for s in slots),
            "types": "{" + (",".join(convert.string2quote(k) + ": " + v.__name__ for k, v in types.items())) + "}",
            "constraint_expr": jx_expression(constraint).to_python(),
            "constraint": value2json(constraint)
        }
    )

    return _exec(code, name)


def _exec(code, name):
    try:
        globs = globals()
        fake_locals = {}
        exec(code, globs, fake_locals)
        temp = globs[name] = fake_locals[name]
        return temp
    except Exception as e:
        Log.error("Can not make class\n{{code}}", code=code, cause=e)


def value2quote(value):
    # RETURN PRETTY PYTHON CODE FOR THE SAME
    if isinstance(value, text_type):
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

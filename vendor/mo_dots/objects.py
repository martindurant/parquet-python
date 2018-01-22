# encoding: utf-8
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

from collections import Mapping
from datetime import date, datetime
from decimal import Decimal

from mo_dots import wrap, unwrap, Data, FlatList, NullType, get_attr, set_attr
from mo_future import text_type, binary_type, get_function_defaults, get_function_arguments, none_type, generator_types

_get = object.__getattribute__
_set = object.__setattr__
WRAPPED_CLASSES = set()


class DataObject(Mapping):
    """
    TREAT AN OBJECT LIKE DATA
    """

    def __init__(self, obj):
        _set(self, "_obj", obj)

    def __getattr__(self, item):
        obj = _get(self, "_obj")
        output = get_attr(obj, item)
        return datawrap(output)

    def __setattr__(self, key, value):
        obj = _get(self, "_obj")
        set_attr(obj, key, value)

    def __getitem__(self, item):
        obj = _get(self, "_obj")
        output = get_attr(obj, item)
        return datawrap(output)

    def keys(self):
        obj = _get(self, "_obj")
        try:
            return obj.__dict__.keys()
        except Exception as e:
            raise e

    def items(self):
        obj = _get(self, "_obj")
        try:
            return obj.__dict__.items()
        except Exception as e:
            return [
                (k, getattr(obj, k, None))
                for k in dir(obj)
                if not k.startswith("__")
            ]

    def iteritems(self):
        obj = _get(self, "_obj")
        try:
            return obj.__dict__.iteritems()
        except Exception as e:
            def output():
                for k in dir(obj):
                    if k.startswith("__"):
                        continue
                    yield k, getattr(obj, k, None)
            return output()

    def __data__(self):
        return self

    def __iter__(self):
        return (k for k in self.keys())

    def __unicode__(self):
        obj = _get(self, "_obj")
        return text_type(obj)

    def __str__(self):
        obj = _get(self, "_obj")
        return str(obj)

    def __len__(self):
        obj = _get(self, "_obj")
        return len(obj)

    def __call__(self, *args, **kwargs):
        obj = _get(self, "_obj")
        return obj(*args, **kwargs)


def datawrap(v):
    type_ = _get(v, "__class__")

    if type_ is dict:
        m = Data()
        _set(m, "_dict", v)  # INJECT m.__dict__=v SO THERE IS NO COPY
        return m
    elif type_ is Data:
        return v
    elif type_ is DataObject:
        return v
    elif type_ is none_type:
        return None   # So we allow `is None`
    elif type_ is list:
        return FlatList(v)
    elif type_ in generator_types:
        return (wrap(vv) for vv in v)
    elif isinstance(v, (text_type, binary_type, int, float, Decimal, datetime, date, Data, FlatList, NullType, none_type)):
        return v
    elif isinstance(v, Mapping):
        return DataObject(v)
    elif hasattr(v, "__data__"):
        return v.__data__()
    else:
        return DataObject(v)


class DictClass(object):
    """
    ALLOW INSTANCES OF class_ TO ACK LIKE dicts
    ALLOW CONSTRUCTOR TO ACCEPT @override
    """

    def __init__(self, class_):
        WRAPPED_CLASSES.add(class_)
        self.class_ = class_
        self.constructor = class_.__init__

    def __call__(self, *args, **kwargs):
        settings = wrap(kwargs).settings

        params = get_function_arguments(self.constructor)[1:]
        func_defaults = get_function_defaults(self.constructor)
        if not func_defaults:
            defaults = {}
        else:
            defaults = {k: v for k, v in zip(reversed(params), reversed(func_defaults))}

        ordered_params = dict(zip(params, args))

        output = self.class_(**params_pack(params, ordered_params, kwargs, settings, defaults))
        return DataObject(output)


def params_pack(params, *args):
    settings = {}
    for a in args:
        for k, v in a.items():
            k = text_type(k)
            if k in settings:
                continue
            settings[k] = v

    output = {str(k): unwrap(settings[k]) for k in params if k in settings}
    return output



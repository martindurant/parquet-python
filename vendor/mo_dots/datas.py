# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

from __future__ import absolute_import, division, unicode_literals

from collections import MutableMapping
from copy import copy, deepcopy
from decimal import Decimal

from mo_future import PY2, generator_types, is_binary, iteritems, long, none_type, text_type

from mo_dots import _getdefault, coalesce, get_logger, hash_value, listwrap, literal_field
from mo_dots.utils import CLASS

_get = object.__getattribute__
_set = object.__setattr__

SLOT = str("_internal_dict")
DEBUG = False


class Data(MutableMapping):
    """
    Please see README.md
    """

    __slots__ = [SLOT]

    def __init__(self, *args, **kwargs):
        """
        CALLING Data(**something) WILL RESULT IN A COPY OF something, WHICH
        IS UNLIKELY TO BE USEFUL. USE wrap() INSTEAD
        """
        if DEBUG:
            d = self._internal_dict
            for k, v in kwargs.items():
                d[literal_field(k)] = unwrap(v)
        else:
            if args:
                args0 = args[0]
                class_ = _get(args0, CLASS)
                if class_ is dict:
                    _set(self, SLOT, args0)
                elif class_ is Data:
                    _set(self, SLOT, _get(args0, SLOT))
                else:
                    _set(self, SLOT, dict(args0))
            elif kwargs:
                _set(self, SLOT, unwrap(kwargs))
            else:
                _set(self, SLOT, {})

    def __bool__(self):
        d = self._internal_dict
        if _get(d, CLASS) is dict:
            return bool(d)
        else:
            return d != None

    def __nonzero__(self):
        d = self._internal_dict
        if _get(d, CLASS) is dict:
            return True if d else False
        else:
            return d != None

    def __contains__(self, item):
        value = Data.__getitem__(self, item)
        if _get(value, CLASS) in data_types or value:
            return True
        return False

    def __iter__(self):
        d = self._internal_dict
        return d.__iter__()

    def __getitem__(self, key):
        if key == None:
            return Null
        if key == ".":
            output = self._internal_dict
            if _get(output, CLASS) in data_types:
                return self
            else:
                return output

        key = text_type(key)
        d = self._internal_dict

        if key.find(".") >= 0:
            seq = _split_field(key)
            for n in seq:
                if _get(d, CLASS) is NullType:
                    d = NullType(d, n)  # OH DEAR, Null TREATS n AS PATH, NOT LITERAL
                elif is_list(d):
                    d = [_getdefault(dd, n) for dd in d]
                else:
                    d = _getdefault(d, n)  # EVERYTHING ELSE TREATS n AS LITERAL

            return wrap(d)
        else:
            o = d.get(key)

        if o == None:
            return NullType(d, key)
        return wrap(o)

    def __setitem__(self, key, value):
        if key == "":
            get_logger().error("key is empty string.  Probably a bad idea")
        if key == None:
            return Null
        if key == ".":
            # SOMETHING TERRIBLE HAPPENS WHEN value IS NOT A Mapping;
            # HOPEFULLY THE ONLY OTHER METHOD RUN ON self IS unwrap()
            v = unwrap(value)
            _set(self, SLOT, v)
            return v

        try:
            d = self._internal_dict
            value = unwrap(value)
            if key.find(".") == -1:
                if value is None:
                    d.pop(key, None)
                else:
                    d[key] = value
                return self

            seq = _split_field(key)
            for k in seq[:-1]:
                d = _getdefault(d, k)
            if value == None:
                try:
                    d.pop(seq[-1], None)
                except Exception as _:
                    pass
            elif d==None:
                d[literal_field(seq[-1])] = value
            else:
                d[seq[-1]] = value
            return self
        except Exception as e:
            raise e

    def __getattr__(self, key):
        d = _get(self, SLOT)
        v = d.get(key)
        t = _get(v, CLASS)

        # OPTIMIZED wrap()
        if t is dict:
            m = object.__new__(Data)
            _set(m, SLOT, v)
            return m
        elif t in (none_type, NullType):
            return NullType(d, key)
        elif t is list:
            return FlatList(v)
        elif t in generator_types:
            return FlatList(list(unwrap(vv) for vv in v))
        else:
            return v

    def __setattr__(self, key, value):
        d = self._internal_dict
        value = unwrap(value)
        if value is None:
            d = self._internal_dict
            d.pop(key, None)
        else:
            d[key] = value
        return self

    def __add__(self, other):
        return _iadd(_iadd({}, self), other)

    def __radd__(self, other):
        return _iadd(_iadd({}, other), self)

    def __iadd__(self, other):
        return _iadd(self, other)


    def __hash__(self):
        d = self._internal_dict
        return hash_value(d)

    def __eq__(self, other):
        if self is other:
            return True

        d = self._internal_dict
        if _get(d, CLASS) is not dict:
            return d == other

        if not d and other == None:
            return False

        if _get(other, CLASS) not in data_types:
            return False
        e = unwrap(other)
        for k, v in d.items():
            if e.get(k) != v:
                return False
        for k, v in e.items():
            if d.get(k) != v:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get(self, key, default=None):
        d = self._internal_dict
        return d.get(key, default)

    def items(self):
        d = self._internal_dict
        return [(k, wrap(v)) for k, v in d.items() if v != None or _get(v, CLASS) in data_types]

    def leaves(self, prefix=None):
        """
        LIKE items() BUT RECURSIVE, AND ONLY FOR THE LEAVES (non dict) VALUES
        """
        return leaves(self, prefix)

    def iteritems(self):
        # LOW LEVEL ITERATION, NO WRAPPING
        d = self._internal_dict
        return ((k, wrap(v)) for k, v in iteritems(d))

    def keys(self):
        d = self._internal_dict
        return set(d.keys())

    def values(self):
        d = self._internal_dict
        return listwrap(list(d.values()))

    def clear(self):
        get_logger().error("clear() not supported")

    def __len__(self):
        d = self._internal_dict
        return dict.__len__(d)

    def copy(self):
        d = self._internal_dict
        if _get(d, CLASS) is dict:
            return Data(**self)
        else:
            return copy(d)

    def __copy__(self):
        d = self._internal_dict
        if _get(d, CLASS) is dict:
            return Data(**self)
        else:
            return copy(d)

    def __deepcopy__(self, memo):
        d = self._internal_dict
        return wrap(deepcopy(d, memo))

    def __delitem__(self, key):
        if key.find(".") == -1:
            d = self._internal_dict
            d.pop(key, None)
            return

        d = self._internal_dict
        seq = _split_field(key)
        for k in seq[:-1]:
            d = d[k]
        d.pop(seq[-1], None)

    def __delattr__(self, key):
        key = text_type(key)
        d = self._internal_dict
        d.pop(key, None)

    def setdefault(self, k, d=None):
        if self[k] == None:
            self[k] = d
        return self

    def __str__(self):
        try:
            return dict.__str__(self._internal_dict)
        except Exception:
            return "{}"

    def __repr__(self):
        try:
            return "Data("+dict.__repr__(self._internal_dict)+")"
        except Exception as e:
            return "Data()"


def leaves(value, prefix=None):
    """
    LIKE items() BUT RECURSIVE, AND ONLY FOR THE LEAVES (non dict) VALUES
    SEE wrap_leaves FOR THE INVERSE

    :param value: THE Mapping TO TRAVERSE
    :param prefix:  OPTIONAL PREFIX GIVEN TO EACH KEY
    :return: Data, WHICH EACH KEY BEING A PATH INTO value TREE
    """
    prefix = coalesce(prefix, "")
    output = []
    for k, v in value.items():
        try:
            if _get(v, CLASS) in data_types:
                output.extend(leaves(v, prefix=prefix + literal_field(k) + "."))
            else:
                output.append((prefix + literal_field(k), unwrap(v)))
        except Exception as e:
            get_logger().error("Do not know how to handle", cause=e)
    return output


def _split_field(field):
    """
    SIMPLE SPLIT, NO CHECKS
    """
    return [k.replace("\a", ".") for k in field.replace("\\.", "\a").split(".")]


class _DictUsingSelf(dict):

    def __init__(self, **kwargs):
        """
        CALLING Data(**something) WILL RESULT IN A COPY OF something, WHICH
        IS UNLIKELY TO BE USEFUL. USE wrap() INSTEAD
        """
        dict.__init__(self)

    def __bool__(self):
        return True

    def __getitem__(self, key):
        if key == None:
            return Null
        if is_binary(key):
            key = key.decode("utf8")

        d=self
        if key.find(".") >= 0:
            seq = _split_field(key)
            for n in seq:
                d = _getdefault(self, n)
            return wrap(d)
        else:
            o = dict.get(d, None)

        if o == None:
            return NullType(d, key)
        return wrap(o)

    def __setitem__(self, key, value):
        if key == "":
            get_logger().error("key is empty string.  Probably a bad idea")
        if isinstance(key, str):
            key = key.decode("utf8")
        d=self
        try:
            value = unwrap(value)
            if key.find(".") == -1:
                if value is None:
                    dict.pop(d, key, None)
                else:
                    dict.__setitem__(d, key, value)
                return self

            seq = _split_field(key)
            for k in seq[:-1]:
                d = _getdefault(d, k)
            if value == None:
                dict.pop(d, seq[-1], None)
            else:
                dict.__setitem__(d, seq[-1], value)
            return self
        except Exception as e:
            raise e

    def __getattr__(self, key):
        if is_binary(key):
            ukey = key.decode("utf8")
        else:
            ukey = key

        d = self
        o = dict.get(d, ukey, None)
        if o == None:
            return NullType(d, ukey)
        return wrap(o)

    def __setattr__(self, key, value):
        if is_binary(key):
            ukey = key.decode("utf8")
        else:
            ukey = key

        d = self
        value = unwrap(value)
        if value is None:
            dict.pop(d, key, None)
        else:
            dict.__setitem__(d, ukey, value)
        return self

    def __hash__(self):
        return hash_value(self)

    def __eq__(self, other):
        if self is other:
            return True

        d = self
        if not d and other == None:
            return True

        if not _get(other, CLASS) in data_types:
            return False
        e = unwrap(other)
        for k, v in dict.items(d):
            if e.get(k) != v:
                return False
        for k, v in e.items():
            if dict.get(d, k, None) != v:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get(self, key, default=None):
        return wrap(dict.get(self, key, default))

    def items(self):
        return [(k, wrap(v)) for k, v in dict.items(self) if v != None or _get(v, CLASS) in data_types]

    def leaves(self, prefix=None):
        """
        LIKE items() BUT RECURSIVE, AND ONLY FOR THE LEAVES (non dict) VALUES
        """
        prefix = coalesce(prefix, "")
        output = []
        for k, v in self.items():
            if _get(v, CLASS) in data_types:
                output.extend(wrap(v).leaves(prefix=prefix + literal_field(k) + "."))
            else:
                output.append((prefix + literal_field(k), v))
        return output

    if PY2:
        def iteritems(self):
            for k, v in dict.iteritems(self):
                yield k, wrap(v)
    else:
        def iteritems(self):
            for k, v in dict.items(self):
                yield k, wrap(v)


    def keys(self):
        return set(dict.keys(self))

    def values(self):
        return listwrap(dict.values(self))

    def clear(self):
        get_logger().error("clear() not supported")

    def __len__(self):
        d = self._internal_dict
        return d.__len__()

    def copy(self):
        return Data(**self)

    def __copy__(self):
        return Data(**self)

    def __deepcopy__(self, memo):
        return wrap(dict.__deepcopy__(self, memo))

    def __delitem__(self, key):
        if is_binary(key):
            key = key.decode("utf8")

        if key.find(".") == -1:
            dict.pop(self, key, None)
            return

        d = self
        seq = _split_field(key)
        for k in seq[:-1]:
            d = d[k]
        d.pop(seq[-1], None)

    def __delattr__(self, key):
        if isinstance(key, str):
            key = key.decode("utf8")

        dict.pop(self, key, None)

    def setdefault(self, k, d=None):
        if self[k] == None:
            self[k] = d
        return self

    def __str__(self):
        try:
            return dict.__str__(self)
        except Exception as e:
            return "{}"

    def __repr__(self):
        try:
            return "Data("+dict.__repr__(self)+")"
        except Exception as e:
            return "Data()"


def _str(value, depth):
    """
    FOR DEBUGGING POSSIBLY RECURSIVE STRUCTURES
    """
    output = []
    if depth >0 and _get(value, CLASS) in data_types:
        for k, v in value.items():
            output.append(str(k) + "=" + _str(v, depth - 1))
        return "{" + ",\n".join(output) + "}"
    elif depth >0 and is_list(value):
        for v in value:
            output.append(_str(v, depth-1))
        return "[" + ",\n".join(output) + "]"
    else:
        return str(type(value))


def _iadd(self, other):
    if not _get(other, CLASS) in data_types:
        get_logger().error("Expecting a Mapping")
    d = unwrap(self)
    for ok, ov in other.items():
        sv = d.get(ok)
        if sv == None:
            d[ok] = deepcopy(ov)
        elif isinstance(ov, (Decimal, float, long, int)):
            if _get(sv, CLASS) in data_types:
                get_logger().error(
                    "can not add {{stype}} with {{otype}",
                    stype=_get(sv, CLASS).__name__,
                    otype=_get(ov, CLASS).__name__
                )
            elif is_list(sv):
                d[ok].append(ov)
            else:
                d[ok] = sv + ov
        elif is_list(ov):
            d[ok] = listwrap(sv) + ov
        elif _get(ov, CLASS) in data_types:
            if _get(sv, CLASS) in data_types:
                _iadd(sv, ov)
            elif is_list(sv):
                d[ok].append(ov)
            else:
                get_logger().error(
                    "can not add {{stype}} with {{otype}",
                    stype=_get(sv, CLASS).__name__,
                    otype=_get(ov, CLASS).__name__
                )
        else:
            if _get(sv, CLASS) in data_types:
                get_logger().error(
                    "can not add {{stype}} with {{otype}",
                    stype=_get(sv, CLASS).__name__,
                    otype=_get(ov, CLASS).__name__
                )
            else:
                d[ok].append(ov)
    return self


data_types = (Data, dict)  # TYPES TO HOLD DATA


def register_data(type_):
    """
    :param type_:  ADD OTHER TYPE THAT HOLDS DATA
    :return:
    """
    global data_types
    data_types = tuple(set(data_types + (type_,)))


def is_data(d):
    """
    :param d:
    :return: True IF d IS A TYPE THAT HOLDS DATA
    """
    return d.__class__ in data_types


from mo_dots.nones import Null, NullType
from mo_dots.lists import is_list, FlatList
from mo_dots import unwrap, wrap

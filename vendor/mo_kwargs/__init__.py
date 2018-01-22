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

from collections import Mapping

from mo_future import text_type, get_function_arguments, get_function_defaults, get_function_name
from mo_dots import zip as dict_zip, get_logger, wrap

from mo_logs import Except


def override(func):
    """
    THIS DECORATOR WILL PUT ALL PARAMETERS INTO THE `kwargs` PARAMETER AND
    PUT ALL `kwargs` PARAMETERS INTO THE FUNCTION PARAMETERS.  THIS HAS BOTH
    THE BENEFIT OF HAVING ALL PARAMETERS IN ONE PLACE (kwargs) AND ALL
    PARAMETERS ARE EXPLICIT FOR CLARITY.

    OF COURSE, THIS MEANS PARAMETER ASSIGNMENT MAY NOT BE UNIQUE: VALUES CAN
    COME FROM EXPLICIT CALL PARAMETERS, OR FROM THE kwargs PARAMETER.  IN
    THESE CASES, PARAMETER VALUES ARE CHOSEN IN THE FOLLOWING ORDER:
    1) EXPLICT CALL PARAMETERS
    2) PARAMETERS FOUND IN kwargs
    3) DEFAULT VALUES ASSIGNED IN FUNCTION DEFINITION
    """

    params = get_function_arguments(func)
    if not get_function_defaults(func):
        defaults = {}
    else:
        defaults = {k: v for k, v in zip(reversed(params), reversed(get_function_defaults(func)))}

    if "kwargs" not in params:
        # WE ASSUME WE ARE ONLY ADDING A kwargs PARAMETER TO SOME REGULAR METHOD
        def w_settings(*args, **kwargs):
            settings = kwargs.get("kwargs")

            params = get_function_arguments(func)
            if not get_function_defaults(func):
                defaults = {}
            else:
                defaults = {k: v for k, v in zip(reversed(params), reversed(get_function_defaults(func)))}

            ordered_params = dict(zip(params, args))

            return func(**params_pack(params, ordered_params, kwargs, settings, defaults))
        return w_settings

    def wrapper(*args, **kwargs):
        try:
            func_name = get_function_name(func)
            if func_name in ("__init__", "__new__") and "kwargs" in kwargs:
                packed = params_pack(params, kwargs, dict_zip(params[1:], args[1:]), kwargs["kwargs"], defaults)
                return func(args[0], **packed)
            elif func_name in ("__init__", "__new__") and len(args) == 2 and len(kwargs) == 0 and isinstance(args[1], Mapping):
                # ASSUME SECOND UNNAMED PARAM IS kwargs
                packed = params_pack(params, args[1], defaults)
                return func(args[0], **packed)
            elif func_name in ("__init__", "__new__"):
                # DO NOT INCLUDE self IN kwargs
                packed = params_pack(params, kwargs, dict_zip(params[1:], args[1:]), defaults)
                return func(args[0], **packed)
            elif params[0] == "self" and "kwargs" in kwargs:
                packed = params_pack(params, kwargs, dict_zip(params[1:], args[1:]), kwargs["kwargs"], defaults)
                return func(args[0], **packed)
            elif params[0] == "self" and len(args) == 2 and len(kwargs) == 0 and isinstance(args[1], Mapping):
                # ASSUME SECOND UNNAMED PARAM IS kwargs
                packed = params_pack(params, args[1], defaults)
                return func(args[0], **packed)
            elif params[0] == "self":
                packed = params_pack(params, kwargs, dict_zip(params[1:], args[1:]), defaults)
                return func(args[0], **packed)
            elif len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], Mapping):
                # ASSUME SINGLE PARAMETER IS A SETTING
                packed = params_pack(params, args[0], defaults)
                return func(**packed)
            elif "kwargs" in kwargs and isinstance(kwargs["kwargs"], Mapping):
                # PUT args INTO kwargs
                packed = params_pack(params, kwargs, dict_zip(params, args), kwargs["kwargs"], defaults)
                return func(**packed)
            else:
                # PULL kwargs OUT INTO PARAMS
                packed = params_pack(params, kwargs, dict_zip(params, args), defaults)
                return func(**packed)
        except TypeError as e:
            e = Except.wrap(e)
            if e.message.startswith(func_name) and "takes at least" in e:
                missing = [p for p in params if str(p) not in packed]
                get_logger().error(
                    "Problem calling {{func_name}}:  Expecting parameter {{missing}}",
                    func_name=func_name,
                    missing=missing,
                    stack_depth=1
                )
            get_logger().error("Error dispatching call", e)
    return wrapper


def params_pack(params, *args):
    settings = {}
    for a in args:
        if a == None:
            continue
        for k, v in a.items():
            k = text_type(k)
            if k in settings:
                continue
            settings[k] = v
    settings["kwargs"] = settings

    output = wrap({str(k): settings[k] for k in params if k in settings})
    return output

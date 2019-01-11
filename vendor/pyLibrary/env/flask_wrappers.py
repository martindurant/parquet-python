# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
import flask
from flask import Response

from mo_dots import coalesce
from mo_files import File
from mo_json import value2json
from mo_logs import Log
from mo_logs.strings import unicode2utf8
from pyLibrary.env.big_data import ibytes2icompressed

TOO_SMALL_TO_COMPRESS = 510  # DO NOT COMPRESS DATA WITH LESS THAN THIS NUMBER OF BYTES


def gzip_wrapper(func, compress_lower_limit=None):
    compress_lower_limit = coalesce(compress_lower_limit, TOO_SMALL_TO_COMPRESS)

    def output(*args, **kwargs):
        response = func(*args, **kwargs)
        accept_encoding = flask.request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response

        response.headers['Content-Encoding'] = 'gzip'
        response.response = ibytes2icompressed(response.response)

        return response

    return output


def cors_wrapper(func):
    """
    Decorator for CORS
    :param func:  Flask method that handles requests and returns a response
    :return: Same, but with permissive CORS headers set
    """
    def _setdefault(obj, key, value):
        if value == None:
            return
        obj.setdefault(key, value)

    def output(*args, **kwargs):
        response = func(*args, **kwargs)
        headers = response.headers
        _setdefault(headers, "Access-Control-Allow-Origin", "*")
        _setdefault(headers, "Access-Control-Allow-Headers", flask.request.headers.get("Access-Control-Request-Headers"))
        _setdefault(headers, "Access-Control-Allow-Methods", flask.request.headers.get("Access-Control-Request-Methods"))
        _setdefault(headers, "Content-Type", "application/json")
        _setdefault(headers, "Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
        return response

    output.provide_automatic_options = False
    output.__name__ = func.__name__
    return output


def dockerflow(flask_app, backend_check):
    """
    ADD ROUTING TO HANDLE DOCKERFLOW APP REQUIREMENTS
    (see https://github.com/mozilla-services/Dockerflow#containerized-app-requirements)
    :param flask_app: THE (Flask) APP
    :param backend_check: METHOD THAT WILL CHECK THE BACKEND IS WORKING AND RAISE AN EXCEPTION IF NOT
    :return:
    """
    global VERSION_JSON

    try:
        VERSION_JSON = File("version.json").read_bytes()

        @cors_wrapper
        def version():
            return Response(
                VERSION_JSON,
                status=200,
                headers={
                    "Content-Type": "application/json"
                }
            )

        @cors_wrapper
        def heartbeat():
            try:
                backend_check()
                return Response(status=200)
            except Exception as e:
                Log.warning("heartbeat failure", cause=e)
                return Response(
                    unicode2utf8(value2json(e)),
                    status=500,
                    headers={
                        "Content-Type": "application/json"
                    }
                )

        @cors_wrapper
        def lbheartbeat():
            return Response(status=200)

        flask_app.add_url_rule(str('/__version__'), None, version, defaults={}, methods=[str('GET'), str('POST')])
        flask_app.add_url_rule(str('/__heartbeat__'), None, heartbeat, defaults={}, methods=[str('GET'), str('POST')])
        flask_app.add_url_rule(str('/__lbheartbeat__'), None, lbheartbeat, defaults={}, methods=[str('GET'), str('POST')])
    except Exception as e:
        Log.error("Problem setting up listeners for dockerflow", cause=e)


VERSION_JSON = None

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
from decimal import Decimal

from mo_dots import wrap
from mo_json import datetime2unix, value2json
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import ALARM, ERROR, NOTE, WARNING
from mo_logs.log_usingElasticSearch import _deep_json_to_string
from mo_logs.log_usingNothing import StructuredLogger


class StructuredLogger_usingMozLog(StructuredLogger):
    """
    WRITE TO MozLog STANDARD FORMAT
    https://wiki.mozilla.org/Firefox/Services/Logging
    """
    @override
    def __init__(self, stream, app_name):
        """
        :param stream: MozLog IS A JSON FORMAT, WHICH IS BYTES
        :param app_name: MozLog WOULD LIKE TO KNOW WHAT APP IS MAKING THESE LOGS
        """
        self.stream = stream
        self.app_name = app_name
        if not app_name:
            Log.error("mozlog expects an `app_name` in the config")
        if not Log.trace:
            Log.error("mozlog expects trace=True so it get s the information it requires")

    def write(self, template, params):
        output = {
            "Timestamp": (Decimal(datetime2unix(params.timestamp)) * Decimal(1e9)).to_integral_exact(),  # NANOSECONDS
            "Type": params.template,
            "Logger": params.machine.name,
            "Hostname": self.app_name,
            "EnvVersion": "2.0",
            "Severity": severity_map.get(params.context, 3),  # https://en.wikipedia.org/wiki/Syslog#Severity_levels
            "Pid": params.machine.pid,
            "Fields": {
                k: _deep_json_to_string(v, 0)
                for k, v in wrap(params).leaves()
            }
        }
        self.stream.write(value2json(output).encode('utf8'))
        self.stream.write(b'\n')


severity_map = {
    ERROR: 3,
    WARNING: 4,
    ALARM: 5,
    NOTE: 6
}


def datatime2decimal(value):
    return

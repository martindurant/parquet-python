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
from mo_dots import wrap
from mo_future import ConfigParser, StringIO


def ini2value(ini_content):
    """
    INI FILE CONTENT TO Data
    """
    buff = StringIO(ini_content)
    config = ConfigParser()
    config._read(buff, "dummy")

    output = {}
    for section in config.sections():
        output[section] = s = {}
        for k, v in config.items(section):
            s[k] = v
    return wrap(output)

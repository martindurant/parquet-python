# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import division, unicode_literals

from mo_future import is_text, is_binary
import datetime

from mo_dots import Null
from mo_future import text_type
from mo_logs import Log


class Version(object):

    __slots__ = ["version"]

    def __new__(cls, version):
        if version == None:
            return Null
        else:
            return object.__new__(cls)

    def __init__(self, version):
        if isinstance(version, tuple):
            self.version = version
        elif isinstance(version, Version):
            self.version = version.version
        else:
            self.version = tuple(map(int, version.split('.')))

        if len(self.version) != 3:
            Log.error("expecting <major>.<minor>.<mini> version format")

    def __gt__(self, other):
        other = Version(other)
        for s, o in zip(self.version, other.version):
            if s < o:
                return False
            elif s > o:
                return True

        return False

    def __ge__(self, other):
        return self == other or self > other

    def __eq__(self, other):
        other = Version(other)
        return self.version == other.version

    def __le__(self, other):
        return self == other or not (self > other)

    def __lt__(self, other):
        return not (self == other) and not (self > other)

    def __ne__(self, other):
        other = Version(other)
        return self.version != other.version

    def __str__(self):
        return text_type(".").join(map(text_type, self.version))

    def __add__(self, other):
        major, minor, mini = self.version
        minor += other
        mini = datetime.datetime.utcnow().strftime("%y%j")
        return Version((major, minor, mini))

    @property
    def major(self):
        return self.version[0]

    @property
    def minor(self):
        return self.version[1]

    @property
    def mini(self):
        return self.version[2]

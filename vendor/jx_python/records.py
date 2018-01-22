# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from mo_dots import listwrap


class Record(object):

    def __init__(self, coord, cube):
        self.coord = coord
        self.cube = cube

    def __getitem__(self, item):
        for s in listwrap(self.cube.select):
            if s.name == item:
                return self.cube.data[item]
        for i, e in enumerate(self.cube.edges):
            if e.name == item:
                return e.domain.partition[self.coord[i]]

    def __getattr__(self, item):
        return self[item]

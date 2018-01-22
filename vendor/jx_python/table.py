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
from mo_dots import Data


class Table(object):

    __slots__ = ['header', 'data', 'meta']


    def __init__(self, header=None, data=None):
        self.header = header

        self.data = data
        self.meta = Data()

    def groupby(self, keys):
        pass



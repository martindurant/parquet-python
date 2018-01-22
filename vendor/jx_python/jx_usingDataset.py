# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import dataset

from jx_python.containers.Table_usingDataset import Table_usingDataset


class Dataset(object):


    def __init__(self):
        self.db = dataset.connect('sqlite:///:memory:')


    def get_or_create_table(self, name, uid):
        return Table_usingDataset(name, self.db, primary_id=uid)




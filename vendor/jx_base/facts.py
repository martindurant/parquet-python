# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http:# mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals


from mo_future import is_text, is_binary
class Facts(object):
    """
    REPRESENT A HIERARCHICAL DATASTORE: MULTIPLE TABLES IN A DATABASE ALONG
    WITH THE RELATIONS THAT CONNECT THEM ALL, BUT LIMITED TO A TREE
    """

    def __init__(self, container, snowflake):
        self.container = container
        self.snowflake = snowflake

    @property
    def namespace(self):
        return self.container.namespace

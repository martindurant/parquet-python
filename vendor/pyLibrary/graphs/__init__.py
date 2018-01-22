# encoding: utf-8
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


class Graph(object):
    def __init__(self, node_type=None):
        self.nodes = []
        self.edges = []
        self.node_type = node_type


    def add_edge(self, edge):
        self.edges.append(edge)

    def remove_children(self, node):
        self.edges = [e for e in self.edges if e[0] != node]

    def get_children(self, node):
        #FIND THE REVISION
        #
        return [c for p, c in self.edges if p == node]

    def get_parents(self, node):
        return [p for p, c in self.edges if c == node]

    def get_edges(self, node):
        return [(p, c) for p, c in self.edges if p == node or c == node]

    def get_family(self, node):
        """
        RETURN ALL ADJACENT NODES
        """
        return set([p if c == node else c for p, c in self.edges])


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

from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import expand_template
from mo_threads import Queue


class StructuredLogger_usingQueue(StructuredLogger):

    def __init__(self, name=None):
        queue_name = "log messages to queue"
        if name:
            queue_name += " "+name
        self.queue = Queue(queue_name)

    def write(self, template, params):
        self.queue.add(expand_template(template, params))

    def stop(self):
        self.queue.close()

    def pop(self):
        lines = self.queue.pop()
        output = []
        for l in lines.split("\n"):
            if l[19:22] == " - ":
                l = l[22:]
            if l.strip().startswith("File"):
                continue
            output.append(l)
        return "\n".join(output).strip()

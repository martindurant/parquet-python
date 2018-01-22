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

import time

from mo_future import allocate_lock
from mo_logs import Log
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import expand_template


class StructuredLogger_usingFile(StructuredLogger):
    def __init__(self, file):
        assert file
        from mo_files import File

        self.file = File(file)
        if self.file.exists:
            self.file.backup()
            self.file.delete()

        self.file_lock = allocate_lock()

    def write(self, template, params):
        try:
            with self.file_lock:
                self.file.append(expand_template(template, params))
        except Exception as e:
            Log.warning("Problem writing to file {{file}}, waiting...", file=file.name, cause=e)
            time.sleep(5)


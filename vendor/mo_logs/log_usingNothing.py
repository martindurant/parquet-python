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
class StructuredLogger(object):
    """
    ABSTRACT BASE CLASS FOR JSON LOGGING
    """
    def write(self, template, params):
        pass

    def stop(self):
        pass




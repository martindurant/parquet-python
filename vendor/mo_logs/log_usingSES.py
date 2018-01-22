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

from boto.ses import connect_to_region

from mo_dots import listwrap, unwrap, get_module, set_default, literal_field, Data, wrap
from mo_kwargs import override
from mo_logs import Log, suppress_exception
from mo_logs.exceptions import ALARM, NOTE
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import expand_template
from mo_threads import Lock
from mo_times import Date, Duration, HOUR, MINUTE


class StructuredLogger_usingSES(StructuredLogger):

    @override
    def __init__(
        self,
        from_address,
        to_address,
        subject,
        region,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        cc=None,
        log_type="ses",
        max_interval=HOUR,
        kwargs=None
    ):
        assert kwargs.log_type == "ses", "Expecing settings to be of type 'ses'"
        self.settings = kwargs
        self.accumulation = []
        self.cc = listwrap(cc)
        self.next_send = Date.now() + MINUTE
        self.locker = Lock()
        self.settings.max_interval = Duration(kwargs.max_interval)

    def write(self, template, params):
        with self.locker:
            if params.context not in [NOTE, ALARM]:  # SEND ONLY THE NOT BORING STUFF
                self.accumulation.append((template, params))

            if Date.now() > self.next_send:
                self._send_email()

    def stop(self):
        with self.locker:
            self._send_email()

    def _send_email(self):
        try:
            if not self.accumulation:
                return
            with Closer(connect_to_region(
                self.settings.region,
                aws_access_key_id=unwrap(self.settings.aws_access_key_id),
                aws_secret_access_key=unwrap(self.settings.aws_secret_access_key)
            )) as conn:

                # WHO ARE WE SENDING TO
                emails = Data()
                for template, params in self.accumulation:
                    content = expand_template(template, params)
                    emails[literal_field(self.settings.to_address)] += [content]
                    for c in self.cc:
                        if any(c in params.params.error for c in c.contains):
                            emails[literal_field(c.to_address)] += [content]

                # SEND TO EACH
                for to_address, content in emails.items():
                    conn.send_email(
                        source=self.settings.from_address,
                        to_addresses=listwrap(to_address),
                        subject=self.settings.subject,
                        body="\n\n".join(content),
                        format="text"
                    )

            self.next_send = Date.now() + self.settings.max_interval
            self.accumulation = []
        except Exception as e:
            self.next_send = Date.now() + self.settings.max_interval
            Log.warning("Could not send", e)



class Closer(object):
    def __init__(self, resource):
        self.resource = resource

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with suppress_exception:
            self.resource.close()

    def __getattr__(self, item):
        return getattr(self.resource, item)

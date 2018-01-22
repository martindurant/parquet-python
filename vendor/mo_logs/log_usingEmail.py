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

from mo_dots import listwrap, get_module, set_default, literal_field, Data
from mo_kwargs import override
from mo_logs import Log
from mo_logs.exceptions import ALARM, NOTE
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import expand_template
from mo_threads import Lock
from mo_times import Date, Duration, HOUR, MINUTE
from pyLibrary.env.emailer import Emailer


class StructuredLogger_usingEmail(StructuredLogger):

    @override
    def __init__(
        self,
        from_address,
        to_address,
        subject,
        host,
        username,
        password,
        port=465,
        use_ssl=1,
        cc=None,
        log_type="email",
        max_interval=HOUR,
        kwargs=None
    ):
        """
        SEND WARNINGS AND ERRORS VIA EMAIL

        settings = {
            "log_type":"email",
            "from_address": "klahnakoski@mozilla.com",
            "to_address": "klahnakoski@mozilla.com",
            "cc":[
                {"to_address":"me@example.com", "where":{"eq":{"template":"gr"}}}
            ],
            "subject": "Problem in Pulse Logger",
            "host": "mail.mozilla.com",
            "port": 465,
            "username": "username",
            "password": "password",
            "use_ssl": 1
        }

        """
        assert kwargs.log_type == "email", "Expecing settings to be of type 'email'"
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
            if self.accumulation:
                with Emailer(self.settings) as emailer:
                    # WHO ARE WE SENDING TO
                    emails = Data()
                    for template, params in self.accumulation:
                        content = expand_template(template, params)
                        emails[literal_field(self.settings.to_address)] += [content]
                        for c in self.cc:
                            if any(d in params.params.error for d in c.contains):
                                emails[literal_field(c.to_address)] += [content]

                    # SEND TO EACH
                    for to_address, content in emails.items():
                        emailer.send_email(
                            from_address=self.settings.from_address,
                            to_address=listwrap(to_address),
                            subject=self.settings.subject,
                            text_data="\n\n".join(content)
                        )

            self.next_send = Date.now() + self.settings.max_interval
            self.accumulation = []
        except Exception as e:
            self.next_send = Date.now() + self.settings.max_interval
            Log.warning("Could not send", e)

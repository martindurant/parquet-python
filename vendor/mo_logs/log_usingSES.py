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
from boto.ses import connect_to_region

from mo_dots import Data, listwrap, literal_field, unwrap
from mo_kwargs import override
from mo_logs import Log, suppress_exception
from mo_logs.exceptions import ALARM, NOTE
from mo_logs.log_usingNothing import StructuredLogger
from mo_logs.strings import expand_template
from mo_math.randoms import Random
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
        average_interval=HOUR,
        kwargs=None
    ):
        """
        SEND WARNINGS AND ERRORS VIA EMAIL

        settings = {
            "log_type": "ses",
            "from_address": "klahnakoski@mozilla.com",
            "to_address": "klahnakoski@mozilla.com",
            "cc":[
                {"to_address":"me@example.com", "where":{"eq":{"template":"gr"}}}
            ],
            "subject": "[ALERT][STAGING] Problem in ETL",
            "aws_access_key_id": "userkey"
            "aws_secret_access_key": "secret"
            "region":"us-west-2"
        }
        """
        assert kwargs.log_type == "ses", "Expecing settings to be of type 'ses'"
        self.settings = kwargs
        self.accumulation = []
        self.cc = listwrap(cc)
        self.next_send = Date.now() + MINUTE
        self.locker = Lock()
        self.settings.average_interval = Duration(kwargs.average_interval)

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
                        source=self.settings.from_address,
                        to_addresses=listwrap(to_address),
                        subject=self.settings.subject,
                        body="\n\n".join(content),
                        format="text"
                    )

            self.accumulation = []
        except Exception as e:
            Log.warning("Could not send", e)
        finally:
            self.next_send = Date.now() + self.settings.average_interval * (2 * Random.float())


class Emailer(object):
    def __init__(self, settings):
        self.resource = connect_to_region(
            settings.region,
            aws_access_key_id=unwrap(settings.aws_access_key_id),
            aws_secret_access_key=unwrap(settings.aws_secret_access_key)
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with suppress_exception:
            self.resource.close()

    def __getattr__(self, item):
        return getattr(self.resource, item)

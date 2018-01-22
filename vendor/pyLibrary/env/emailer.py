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

import smtplib
import sys

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from mo_logs import Log
from mo_dots import listwrap
from mo_dots import coalesce
from mo_kwargs import override


class Emailer:
    @override
    def __init__(
        self,
        from_address,
        to_address,
        host,
        username,
        password,
        subject="catchy title",
        port=465,
        use_ssl=1,
        kwargs=None
    ):
        self.settings = kwargs
        self.server = None

    def __enter__(self):
        if self.server is not None:
            Log.error("Got a problem")

        if self.settings.use_ssl:
            self.server = smtplib.SMTP_SSL(self.settings.host, self.settings.port)
        else:
            self.server = smtplib.SMTP(self.settings.host, self.settings.port)

        if self.settings.username and self.settings.password:
            self.server.login(self.settings.username, self.settings.password)

        return self

    def __exit__(self, type, value, traceback):
        try:
            self.server.quit()
        except Exception as e:
            Log.warning("Problem with smtp server quit(), ignoring problem", e)

        self.server = None

    def send_email(self,
            from_address=None,
            to_address=None,
            subject=None,
            text_data=None,
            html_data=None
    ):
        """Sends an email.

        from_addr is an email address; to_addrs is a list of email adresses.
        Addresses can be plain (e.g. "jsmith@example.com") or with real names
        (e.g. "John Smith <jsmith@example.com>").

        text_data and html_data are both strings.  You can specify one or both.
        If you specify both, the email will be sent as a MIME multipart
        alternative, i.e., the recipient will see the HTML content if his
        viewer supports it; otherwise he'll see the text content.
        """

        settings = self.settings

        from_address = coalesce(from_address, settings["from"], settings.from_address)
        to_address = listwrap(coalesce(to_address, settings.to_address, settings.to_addrs))

        if not from_address or not to_address:
            raise Exception("Both from_addr and to_addrs must be specified")
        if not text_data and not html_data:
            raise Exception("Must specify either text_data or html_data")

        if not html_data:
            msg = MIMEText(text_data)
        elif not text_data:
            msg = MIMEText(html_data, 'html')
        else:
            msg = MIMEMultipart('alternative')
            msg.attach(MIMEText(text_data, 'plain'))
            msg.attach(MIMEText(html_data, 'html'))

        msg['Subject'] = coalesce(subject, settings.subject)
        msg['From'] = from_address
        msg['To'] = ', '.join(to_address)

        if self.server:
            # CALL AS PART OF A SMTP SESSION
            self.server.sendmail(from_address, to_address, msg.as_string())
        else:
            # CALL AS STAND-ALONE
            with self:
                self.server.sendmail(from_address, to_address, msg.as_string())



if sys.hexversion < 0x020603f0:
    # versions earlier than 2.6.3 have a bug in smtplib when sending over SSL:
    #     http://bugs.python.org/issue4066

    # Unfortunately the stock version of Python in Snow Leopard is 2.6.1, so
    # we patch it here to avoid having to install an updated Python version.
    import socket
    import ssl

    def _get_socket_fixed(self, host, port, timeout):
        if self.debuglevel > 0:
            print>> sys.stderr, 'connect:', (host, port)
        new_socket = socket.create_connection((host, port), timeout)
        new_socket = ssl.wrap_socket(new_socket, self.keyfile, self.certfile)
        self.file = smtplib.SSLFakeFile(new_socket)
        return new_socket

    smtplib.SMTP_SSL._get_socket = _get_socket_fixed



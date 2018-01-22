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

import mo_threads
from mo_logs.exceptions import suppress_exception
from pyLibrary.meta import cache
from mo_threads import Process


@cache
def get_git_revision():
    """
    GET THE CURRENT GIT REVISION
    """
    proc = Process("git log", ["git", "log", "-1"])

    try:
        while True:
            line = proc.stdout.pop().strip().decode('utf8')
            if not line:
                continue
            if line.startswith("commit "):
                return line[7:]
    finally:
        with suppress_exception:
            proc.join()

@cache
def get_remote_revision(url, branch):
    """
    GET REVISION OF A REMOTE BRANCH
    """

    mo_threads.DEBUG = True
    proc = Process("git remote revision", ["git", "ls-remote", url, "refs/heads/" + branch])

    try:
        while True:
            raw_line = proc.stdout.pop()
            line = raw_line.strip().decode('utf8')
            if not line:
                continue
            return line.split("\t")[0]
    finally:
        try:
            proc.join()
        except Exception:
            pass

    return None


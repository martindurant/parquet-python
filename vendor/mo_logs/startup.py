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
import argparse as _argparse
import os
import sys
import tempfile

from mo_dots import coalesce, listwrap, unwrap, wrap
from mo_files import File
import mo_json_config
from mo_logs import Log


# PARAMETERS MATCH argparse.ArgumentParser.add_argument()
# https://docs.python.org/dev/library/argparse.html#the-add-argument-method
#
# name or flags - Either a name or a list of option strings, e.g. foo or -f, --foo.
# action - The basic type of action to be taken when this argument is encountered at the command line.
# nargs - The number of command-line arguments that should be consumed.
# const - A constant value required by some action and nargs selections.
# default - The value produced if the argument is absent from the command line.
# type - The type to which the command-line argument should be converted.
# choices - A container of the allowable values for the argument.
# required - Whether or not the command-line option may be omitted (optionals only).
# help - A brief description of what the argument does.
# metavar - A name for the argument in usage messages.
# dest - The name of the attribute to be added to the object returned by parse_args().


class _ArgParser(_argparse.ArgumentParser):
    def error(self, message):
        Log.error("argparse error: {{error}}", error=message)


def argparse(defs):
    parser = _ArgParser()
    for d in listwrap(defs):
        args = d.copy()
        name = args.name
        args.name = None
        parser.add_argument(*unwrap(listwrap(name)), **args)
    namespace, unknown = parser.parse_known_args()
    if unknown:
        Log.warning("Ignoring arguments: {{unknown|json}}", unknown=unknown)
    output = {k: getattr(namespace, k) for k in vars(namespace)}
    return wrap(output)


def read_settings(filename=None, defs=None):
    """
    :param filename: Force load a file
    :param defs: arguments you want to accept
    :param default_filename: A config file from an environment variable (a fallback config file, if no other provided)
    :return:
    """
    # READ SETTINGS
    defs = listwrap(defs)
    defs.append({
        "name": ["--config", "--settings", "--settings-file", "--settings_file"],
        "help": "path to JSON file with settings",
        "type": str,
        "dest": "filename",
        "default": None,
        "required": False
    })
    args = argparse(defs)

    args.filename = coalesce(filename, args.filename, "./config.json")
    settings_file = File(args.filename)
    if not settings_file.exists:
        Log.error("Can not read configuration file {{filename}}", {
            "filename": settings_file.abspath
        })
    settings = mo_json_config.get_file(settings_file)
    settings.args = args
    return settings


# snagged from https://github.com/pycontribs/tendo/blob/master/tendo/singleton.py (under licence PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2)
class SingleInstance:
    """
    ONLY ONE INSTANCE OF PROGRAM ALLOWED
    If you want to prevent your script from running in parallel just instantiate SingleInstance() class.
    If is there another instance already running it will exist the application with the message
    "Another instance is already running, quitting.", returning -1 error code.

    with SingleInstance():
        <your code here>

    settings = startup.read_settings()
    with SingleInstance(settings.args.filename):
        <your code here>

    This option is very useful if you have scripts executed by crontab at small amounts of time.

    Remember that this works by creating a lock file with a filename based on the full path to the script file.
    """
    def __init__(self, flavor_id=""):
        self.initialized = False
        appname = os.path.splitext(os.path.abspath(sys.argv[0]))[0]
        basename = ((appname + '-%s') % flavor_id).replace("/", "-").replace(":", "").replace("\\", "-").replace("-.-", "-") + '.lock'
        self.lockfile = os.path.normpath(tempfile.gettempdir() + '/' + basename)


    def __enter__(self):
        Log.note("SingleInstance.lockfile = " + self.lockfile)
        if sys.platform == 'win32':
            try:
                # file already exists, we try to remove (in case previous execution was interrupted)
                if os.path.exists(self.lockfile):
                    os.unlink(self.lockfile)
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except Exception as e:
                Log.alarm("Another instance is already running, quitting.")
                sys.exit(-1)
        else: # non Windows
            import fcntl
            self.fp = open(self.lockfile, 'w')
            try:
                fcntl.lockf(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except IOError:
                Log.alarm("Another instance is already running, quitting.")
                sys.exit(-1)
        self.initialized = True

    def __exit__(self, type, value, traceback):
        self.__del__()

    def __del__(self):
        temp, self.initialized = self.initialized, False
        if not temp:
            return
        try:
            if sys.platform == 'win32':
                if hasattr(self, 'fd'):
                    os.close(self.fd)
                    os.unlink(self.lockfile)
            else:
                import fcntl
                fcntl.lockf(self.fp, fcntl.LOCK_UN)
                if os.path.isfile(self.lockfile):
                    os.unlink(self.lockfile)
        except Exception as e:
            Log.warning("Problem with SingleInstance __del__()", e)
            sys.exit(-1)


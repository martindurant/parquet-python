# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
import os
import subprocess

from mo_dots import NullType, set_default
from mo_future import none_type
from mo_logs import Log, strings
from mo_logs.exceptions import Except
from mo_threads.lock import Lock
from mo_threads.queues import Queue
from mo_threads.signal import Signal
from mo_threads.threads import THREAD_STOP, Thread
from mo_threads.till import Till

DEBUG = False


class Process(object):
    def __init__(self, name, params, cwd=None, env=None, debug=False, shell=False, bufsize=-1):
        self.name = name
        self.service_stopped = Signal("stopped signal for " + strings.quote(name))
        self.stdin = Queue("stdin for process " + strings.quote(name), silent=True)
        self.stdout = Queue("stdout for process " + strings.quote(name), silent=True)
        self.stderr = Queue("stderr for process " + strings.quote(name), silent=True)

        try:
            self.debug = debug or DEBUG
            self.service = service = subprocess.Popen(
                [str(p) for p in params],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=bufsize,
                cwd=cwd if isinstance(cwd, (str, NullType, none_type)) else cwd.abspath,
                env={str(k): str(v) for k, v in set_default(env, os.environ).items()},
                shell=shell
            )

            self.please_stop = Signal()
            self.please_stop.on_go(self._kill)
            self.thread_locker = Lock()
            self.children = [
                Thread.run(self.name + " stdin", self._writer, service.stdin, self.stdin, please_stop=self.service_stopped, parent_thread=self),
                Thread.run(self.name + " stdout", self._reader, "stdout", service.stdout, self.stdout, please_stop=self.service_stopped, parent_thread=self),
                Thread.run(self.name + " stderr", self._reader, "stderr", service.stderr, self.stderr, please_stop=self.service_stopped, parent_thread=self),
                Thread.run(self.name + " waiter", self._monitor, parent_thread=self),
            ]
        except Exception as e:
            Log.error("Can not call", e)

        self.debug and Log.note("{{process}} START: {{command}}", process=self.name, command=" ".join(map(strings.quote, params)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join(raise_on_error=True)

    def stop(self):
        self.stdin.add(THREAD_STOP)  # ONE MORE SEND
        self.please_stop.go()

    def join(self, raise_on_error=False):
        self.service_stopped.wait()
        with self.thread_locker:
            child_threads, self.children = self.children, []
        for c in child_threads:
            c.join()
        if raise_on_error and self.returncode != 0:
            Log.error(
                "{{process}} FAIL: returncode={{code}}\n{{stderr}}",
                process=self.name,
                code=self.service.returncode,
                stderr=list(self.stderr)
            )
        return self

    def remove_child(self, child):
        with self.thread_locker:
            try:
                self.children.remove(child)
            except Exception:
                pass

    @property
    def pid(self):
        return self.service.pid

    @property
    def returncode(self):
        return self.service.returncode

    def _monitor(self, please_stop):
        self.service.wait()
        self.debug and Log.note("{{process}} STOP: returncode={{returncode}}", process=self.name, returncode=self.service.returncode)
        self.service_stopped.go()
        please_stop.go()

    def _reader(self, name, pipe, receive, please_stop):
        try:
            while not please_stop and self.service.returncode is None:
                line = pipe.readline().rstrip()
                if line:
                    receive.add(line)
                    self.debug and Log.note("{{process}} ({{name}}): {{line}}", name=name, process=self.name, line=line)
                    continue

                # GRAB A FEW MORE LINES
                for _ in range(100):
                    try:
                        line = pipe.readline().rstrip()
                        if line:
                            receive.add(line)
                            self.debug and Log.note("{{process}} ({{name}}): {{line}}", name=name, process=self.name, line=line)
                            break
                    except Exception:
                        break
                else:
                    Till(seconds=5).wait()

            # GRAB A FEW MORE LINES
            max = 100
            while max:
                try:
                    line = pipe.readline().rstrip()
                    if line:
                        max = 100
                        receive.add(line)
                        self.debug and Log.note("{{process}} ({{name}}): {{line}}", name=name, process=self.name, line=line)
                    else:
                        max -= 1
                except Exception:
                    break
        finally:
            pipe.close()
        self.debug and Log.note("{{process}} ({{name}} is closed)", name=name, process=self.name)

        receive.add(THREAD_STOP)

    def _writer(self, pipe, send, please_stop):
        while not please_stop:
            line = send.pop(till=please_stop)
            if line == THREAD_STOP:
                please_stop.go()
                break

            if line:
                self.debug and Log.note("{{process}} (stdin): {{line}}", process=self.name, line=line.rstrip())
                pipe.write(line.encode('utf8') + b"\n")
                pipe.flush()

    def _kill(self):
        try:
            self.service.kill()
            Log.note("Service was successfully terminated.")
        except Exception as e:
            ee = Except.wrap(e)
            if 'The operation completed successfully' in ee:
                return
            if 'No such process' in ee:
                return

            Log.warning("Failure to kill process {{process|quote}}", process=self.name, cause=ee)



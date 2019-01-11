# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
# THIS THREADING MODULE IS PERMEATED BY THE please_stop SIGNAL.
# THIS SIGNAL IS IMPORTANT FOR PROPER SIGNALLING WHICH ALLOWS
# FOR FAST AND PREDICTABLE SHUTDOWN AND CLEANUP OF THREADS

from __future__ import absolute_import, division, unicode_literals

from mo_future import is_text, is_binary
from copy import copy
from datetime import datetime, timedelta
import signal as _signal
import sys
from time import sleep

from mo_dots import Data, coalesce, unwraplist
from mo_future import allocate_lock, get_function_name, get_ident, start_new_thread, text_type
from mo_logs import Except, Log
from mo_threads.lock import Lock
from mo_threads.profiles import CProfiler, write_profiles
from mo_threads.signal import AndSignals, Signal
from mo_threads.till import Till

DEBUG = False

MAX_DATETIME = datetime(2286, 11, 20, 17, 46, 39)
DEFAULT_WAIT_TIME = timedelta(minutes=10)
THREAD_STOP = "stop"
THREAD_TIMEOUT = "TIMEOUT"

datetime.strptime('2012-01-01', '%Y-%m-%d')  # http://bugs.python.org/issue7980


class AllThread(object):
    """
    RUN ALL ADDED FUNCTIONS IN PARALLEL, BE SURE TO HAVE JOINED BEFORE EXIT
    """

    def __init__(self):
        self.threads = []

    def __enter__(self):
        return self

    # WAIT FOR ALL QUEUED WORK TO BE DONE BEFORE RETURNING
    def __exit__(self, type, value, traceback):
        self.join()

    def join(self):
        exceptions = []
        for t in self.threads:
            try:
                t.join()
            except Exception as e:
                exceptions.append(e)

        if exceptions:
            Log.error("Problem in child threads", cause=exceptions)


    def add(self, target, *args, **kwargs):
        """
        target IS THE FUNCTION TO EXECUTE IN THE THREAD
        """
        t = Thread.run(target.__name__, target, *args, **kwargs)
        self.threads.append(t)


class BaseThread(object):
    __slots__ = ["id", "name", "children", "child_lock", "cprofiler"]

    def __init__(self, ident):
        self.id = ident
        if ident != -1:
            self.name = "Unknown Thread " + text_type(ident)
        self.child_lock = allocate_lock()
        self.children = []
        self.cprofiler = None

    def add_child(self, child):
        with self.child_lock:
            self.children.append(child)

    def remove_child(self, child):
        try:
            with self.child_lock:
                self.children.remove(child)
        except Exception:
            pass


class MainThread(BaseThread):
    def __init__(self):
        BaseThread.__init__(self, get_ident())
        self.name = "Main Thread"
        self.please_stop = Signal()
        self.stop_logging = Log.stop
        self.timers = None

    def stop(self):
        """
        BLOCKS UNTIL ALL THREADS HAVE STOPPED
        THEN RUNS sys.exit(0)
        """
        global DEBUG

        self_thread = Thread.current()
        if self_thread != MAIN_THREAD or self_thread != self:
            Log.error("Only the main thread can call stop() on main thread")

        DEBUG = True
        self.please_stop.go()

        join_errors = []
        with self.child_lock:
            children = copy(self.children)
        for c in reversed(children):
            DEBUG and c.name and Log.note("Stopping thread {{name|quote}}", name=c.name)
            try:
                c.stop()
            except Exception as e:
                join_errors.append(e)

        for c in children:
            DEBUG and c.name and Log.note("Joining on thread {{name|quote}}", name=c.name)
            try:
                c.join()
            except Exception as e:
                join_errors.append(e)

            DEBUG and c.name and Log.note("Done join on thread {{name|quote}}", name=c.name)

        if join_errors:
            Log.error("Problem while stopping {{name|quote}}", name=self.name, cause=unwraplist(join_errors))

        self.stop_logging()
        self.timers.stop()
        self.timers.join()

        write_profiles(self.cprofiler)
        DEBUG and Log.note("Thread {{name|quote}} now stopped", name=self.name)
        sys.exit()

    def wait_for_shutdown_signal(
        self,
        please_stop=False,  # ASSIGN SIGNAL TO STOP EARLY
        allow_exit=False,  # ALLOW "exit" COMMAND ON CONSOLE TO ALSO STOP THE APP
        wait_forever=True  # IGNORE CHILD THREADS, NEVER EXIT.  False => IF NO CHILD THREADS LEFT, THEN EXIT
    ):
        """
        FOR USE BY PROCESSES THAT NEVER DIE UNLESS EXTERNAL SHUTDOWN IS REQUESTED

        CALLING THREAD WILL SLEEP UNTIL keyboard interrupt, OR please_stop, OR "exit"

        :param please_stop:
        :param allow_exit:
        :param wait_forever:: Assume all needed threads have been launched. When done
        :return:
        """
        self_thread = Thread.current()
        if self_thread != MAIN_THREAD or self_thread != self:
            Log.error("Only the main thread can sleep forever (waiting for KeyboardInterrupt)")

        if isinstance(please_stop, Signal):
            # MUTUAL SIGNALING MAKES THESE TWO EFFECTIVELY THE SAME SIGNAL
            self.please_stop.on_go(please_stop.go)
            please_stop.on_go(self.please_stop.go)
        else:
            please_stop = self.please_stop

        if not wait_forever:
            # TRIGGER SIGNAL WHEN ALL CHILDREN THEADS ARE DONE
            with self_thread.child_lock:
                pending = copy(self_thread.children)
            children_done = AndSignals(please_stop, len(pending))
            children_done.signal.on_go(self.please_stop.go)
            for p in pending:
                p.stopped.on_go(children_done.done)

        try:
            if allow_exit:
                _wait_for_exit(please_stop)
            else:
                _wait_for_interrupt(please_stop)
        except KeyboardInterrupt as _:
            Log.alert("SIGINT Detected!  Stopping...")
        except SystemExit as _:
            Log.alert("SIGTERM Detected!  Stopping...")
        finally:
            self.stop()


class Thread(BaseThread):
    """
    join() ENHANCED TO ALLOW CAPTURE OF CTRL-C, AND RETURN POSSIBLE THREAD EXCEPTIONS
    run() ENHANCED TO CAPTURE EXCEPTIONS
    """

    num_threads = 0

    def __init__(self, name, target, *args, **kwargs):
        BaseThread.__init__(self, -1)
        self.name = coalesce(name, "thread_" + text_type(object.__hash__(self)))
        self.target = target
        self.end_of_thread = Data()
        self.synch_lock = Lock("response synch lock")
        self.args = args

        # ENSURE THERE IS A SHARED please_stop SIGNAL
        self.kwargs = copy(kwargs)
        self.kwargs["please_stop"] = self.kwargs.get("please_stop", Signal("please_stop for " + self.name))
        self.please_stop = self.kwargs["please_stop"]

        self.thread = None
        self.stopped = Signal("stopped signal for " + self.name)

        if "parent_thread" in kwargs:
            del self.kwargs["parent_thread"]
            self.parent = kwargs["parent_thread"]
        else:
            self.parent = Thread.current()
            self.parent.add_child(self)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if isinstance(type, BaseException):
            self.please_stop.go()

        # TODO: AFTER A WHILE START KILLING THREAD
        self.join()
        self.args = None
        self.kwargs = None

    def start(self):
        try:
            self.thread = start_new_thread(Thread._run, (self,))
            return self
        except Exception as e:
            Log.error("Can not start thread", e)

    def stop(self):
        """
        SEND STOP SIGNAL, DO NOT BLOCK
        """
        with self.child_lock:
            children = copy(self.children)
        for c in children:
            DEBUG and c.name and Log.note("Stopping thread {{name|quote}}", name=c.name)
            c.stop()
        self.please_stop.go()

        DEBUG and Log.note("Thread {{name|quote}} got request to stop", name=self.name)

    def _run(self):
        self.id = get_ident()
        with RegisterThread(self):
            try:
                if self.target is not None:
                    a, k, self.args, self.kwargs = self.args, self.kwargs, None, None
                    self.end_of_thread.response = self.target(*a, **k)
                    self.parent.remove_child(self)  # IF THREAD ENDS OK, THEN FORGET ABOUT IT
            except Exception as e:
                e = Except.wrap(e)
                with self.synch_lock:
                    self.end_of_thread.exception = e
                with self.parent.child_lock:
                    emit_problem = self not in self.parent.children
                if emit_problem:
                    # THREAD FAILURES ARE A PROBLEM ONLY IF NO ONE WILL BE JOINING WITH IT
                    try:
                        Log.fatal("Problem in thread {{name|quote}}", name=self.name, cause=e)
                    except Exception:
                        sys.stderr.write(str("ERROR in thread: " + self.name + " " + text_type(e) + "\n"))
            finally:
                try:
                    with self.child_lock:
                        children = copy(self.children)
                    for c in children:
                        try:
                            DEBUG and sys.stdout.write(str("Stopping thread " + c.name + "\n"))
                            c.stop()
                        except Exception as e:
                            Log.warning("Problem stopping thread {{thread}}", thread=c.name, cause=e)

                    for c in children:
                        try:
                            DEBUG and sys.stdout.write(str("Joining on thread " + c.name + "\n"))
                            c.join()
                        except Exception as e:
                            Log.warning("Problem joining thread {{thread}}", thread=c.name, cause=e)
                        finally:
                            DEBUG and sys.stdout.write(str("Joined on thread " + c.name + "\n"))

                    del self.target, self.args, self.kwargs
                    DEBUG and Log.note("thread {{name|quote}} stopping", name=self.name)
                except Exception as e:
                    DEBUG and Log.warning("problem with thread {{name|quote}}", cause=e, name=self.name)
                finally:
                    self.stopped.go()
                    DEBUG and Log.note("thread {{name|quote}} is done", name=self.name)

    def is_alive(self):
        return not self.stopped

    def join(self, till=None):
        """
        RETURN THE RESULT {"response":r, "exception":e} OF THE THREAD EXECUTION (INCLUDING EXCEPTION, IF EXISTS)
        """
        if self is Thread:
            Log.error("Thread.join() is not a valid call, use t.join()")

        with self.child_lock:
            children = copy(self.children)
        for c in children:
            c.join(till=till)

        DEBUG and Log.note("{{parent|quote}} waiting on thread {{child|quote}}", parent=Thread.current().name, child=self.name)
        (self.stopped | till).wait()
        if self.stopped:
            self.parent.remove_child(self)
            if not self.end_of_thread.exception:
                return self.end_of_thread.response
            else:
                Log.error("Thread {{name|quote}} did not end well", name=self.name, cause=self.end_of_thread.exception)
        else:
            raise Except(context=THREAD_TIMEOUT)

    @staticmethod
    def run(name, target, *args, **kwargs):
        # ENSURE target HAS please_stop ARGUMENT
        if get_function_name(target) == 'wrapper':
            pass  # GIVE THE override DECORATOR A PASS
        elif "please_stop" not in target.__code__.co_varnames:
            Log.error("function must have please_stop argument for signalling emergency shutdown")

        Thread.num_threads += 1

        output = Thread(name, target, *args, **kwargs)
        output.start()
        return output

    @staticmethod
    def current():
        ident = get_ident()
        with ALL_LOCK:
            output = ALL.get(ident)

        if output is None:
            thread = BaseThread(ident)
            thread.cprofiler = CProfiler()
            thread.cprofiler.__enter__()
            with ALL_LOCK:
                ALL[ident] = thread
            Log.warning("this thread is not known. Register this thread at earliest known entry point.")
            return thread
        return output


class RegisterThread(object):

    def __init__(self, thread=None):
        if thread is None:
            thread = BaseThread(get_ident())
        self.thread = thread

    def __enter__(self):
        with ALL_LOCK:
            ALL[self.thread.id] = self.thread
        self.thread.cprofiler = CProfiler()
        self.thread.cprofiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.cprofiler.__exit__(exc_type, exc_val, exc_tb)
        with ALL_LOCK:
            del ALL[self.thread.id]


def stop_main_thread(*args):
    """
    CLEAN OF ALL THREADS CREATED WITH THIS LIBRARY
    """
    try:
        if len(args) and args[0] != _signal.SIGTERM:
            Log.warning("exit with {{value}}", value=_describe_exit_codes.get(args[0], args[0]))
    except Exception as _:
        pass
    finally:
        MAIN_THREAD.stop()


_describe_exit_codes = {
    _signal.SIGTERM: "SIGTERM",
    _signal.SIGINT: "SIGINT"
}

_signal.signal(_signal.SIGTERM, stop_main_thread)
_signal.signal(_signal.SIGINT, stop_main_thread)


def _wait_for_exit(please_stop):
    """
    /dev/null PIPED TO sys.stdin SPEWS INFINITE LINES, DO NOT POLL AS OFTEN
    """
    try:
        import msvcrt
        _wait_for_exit_on_windows(please_stop)
    except:
        pass

    cr_count = 0  # COUNT NUMBER OF BLANK LINES

    while not please_stop:
        # DEBUG and Log.note("inside wait-for-shutdown loop")
        if cr_count > 30:
            (Till(seconds=3) | please_stop).wait()
        try:
            line = sys.stdin.readline()
        except Exception as e:
            Except.wrap(e)
            if "Bad file descriptor" in e:
                _wait_for_interrupt(please_stop)
                break

        # DEBUG and Log.note("read line {{line|quote}}, count={{count}}", line=line, count=cr_count)
        if line == "":
            cr_count += 1
        else:
            cr_count = -1000000  # NOT /dev/null

        if line.strip() == "exit":
            Log.alert("'exit' Detected!  Stopping...")
            return


def _wait_for_exit_on_windows(please_stop):
    import msvcrt

    line = ""
    while not please_stop:
        if msvcrt.kbhit():
            chr = msvcrt.getche()
            if ord(chr) == 13:
                if line == "exit":
                    Log.alert("'exit' Detected!  Stopping...")
                    return
            elif ord(chr) > 32:
                line += chr
        else:
            sleep(1)



def _wait_for_interrupt(please_stop):
    DEBUG and Log.note("inside wait-for-shutdown loop")
    while not please_stop:
        try:
            sleep(1)
        except Exception:
            pass


MAIN_THREAD = MainThread()

ALL_LOCK = allocate_lock()
ALL = dict()
ALL[get_ident()] = MAIN_THREAD


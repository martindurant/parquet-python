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

import requests
from boto import sqs
from boto import utils as boto_utils
from boto.sqs.message import Message
from mo_dots import wrap, unwrap, coalesce
from mo_json import value2json
from mo_kwargs import override
from mo_logs import Log, machine_metadata
from mo_math import Math
from mo_threads import Thread

import mo_json
from mo_logs.exceptions import Except, suppress_exception
from mo_threads.signal import Signal
from mo_threads.till import Till
from mo_times.durations import SECOND, Duration


class Queue(object):
    @override
    def __init__(
        self,
        name,
        region,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        debug=False,
        kwargs=None
    ):
        self.settings = kwargs
        self.pending = []

        if kwargs.region not in [r.name for r in sqs.regions()]:
            Log.error("Can not find region {{region}} in {{regions}}", region=kwargs.region, regions=[r.name for r in sqs.regions()])

        conn = sqs.connect_to_region(
            region_name=unwrap(kwargs.region),
            aws_access_key_id=unwrap(kwargs.aws_access_key_id),
            aws_secret_access_key=unwrap(kwargs.aws_secret_access_key),
        )
        self.queue = conn.get_queue(kwargs.name)
        if self.queue == None:
            Log.error("Can not find queue with name {{queue}} in region {{region}}", queue=kwargs.name, region=kwargs.region)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        attrib = self.queue.get_attributes("ApproximateNumberOfMessages")
        return int(attrib['ApproximateNumberOfMessages'])

    def add(self, message):
        message = wrap(message)
        m = Message()
        m.set_body(value2json(message))
        self.queue.write(m)

    @property
    def name(self):
        return self.settings.name

    def extend(self, messages):
        for m in messages:
            self.add(m)

    def pop(self, wait=SECOND, till=None):
        if till is not None and not isinstance(till, Signal):
            Log.error("Expecting a signal")

        m = self.queue.read(wait_time_seconds=Math.floor(wait.seconds))
        if not m:
            return None

        self.pending.append(m)
        output = mo_json.json2value(m.get_body())
        return output

    def pop_message(self, wait=SECOND, till=None):
        """
        RETURN TUPLE (message, payload) CALLER IS RESPONSIBLE FOR CALLING message.delete() WHEN DONE
        """
        if till is not None and not isinstance(till, Signal):
            Log.error("Expecting a signal")

        message = self.queue.read(wait_time_seconds=Math.floor(wait.seconds))
        if not message:
            return None
        message.delete = lambda: self.queue.delete_message(message)

        payload = mo_json.json2value(message.get_body())
        return message, payload

    def commit(self):
        pending = self.pending
        self.pending = []
        for p in pending:
            self.queue.delete_message(p)

    def rollback(self):
        if self.pending:
            pending, self.pending = self.pending, []

            for p in pending:
                m = Message()
                m.set_body(p.get_body())
                self.queue.write(m)

            for p in pending:
                self.queue.delete_message(p)

            if self.settings.debug:
                Log.alert("{{num}} messages returned to queue", num=len(pending))

    def close(self):
        self.commit()


def capture_termination_signal(please_stop):
    """
    WILL SIGNAL please_stop WHEN THIS AWS INSTANCE IS DUE FOR SHUTDOWN
    """
    def worker(please_stop):
        while not please_stop:
            try:
                response = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time")
                if response.status_code not in [400, 404]:
                    Log.alert("Shutdown AWS Spot Node {{name}} {{type}}", name=machine_metadata.name, type=machine_metadata.aws_instance_type)
                    please_stop.go()
            except Exception as e:
                e = Except.wrap(e)
                if "Failed to establish a new connection: [Errno 10060]" in e or "A socket operation was attempted to an unreachable network" in e:
                    Log.note("AWS Spot Detection has shutdown, probably not a spot node, (http://169.254.169.254 is unreachable)")
                    return
                else:
                    Log.warning("AWS shutdown detection has problems", cause=e)
                (Till(seconds=61) | please_stop).wait()
            (Till(seconds=11) | please_stop).wait()

    Thread.run("listen for termination", worker)


def get_instance_metadata(timeout=None):
    if not isinstance(timeout, (int, float)):
        timeout = Duration(timeout).seconds

    output = wrap({k.replace("-", "_"): v for k, v in boto_utils.get_instance_metadata(timeout=coalesce(timeout, 5), num_retries=2).items()})
    return output


def aws_retry(func):
    def output(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                e = Except.wrap(e)
                if "Request limit exceeded" in e:
                    Log.warning("AWS Problem", cause=e)
                    continue
                else:
                    Log.error("Problem with call to AWS", cause=e)
    return output


# GET FROM AWS, IF WE CAN
def _get_metadata_from_from_aws(please_stop):
    with suppress_exception:
        ec2 = get_instance_metadata()
        if ec2:
            machine_metadata.aws_instance_type = ec2.instance_type
            machine_metadata.name = ec2.instance_id

Thread.run("get aws machine metadata", _get_metadata_from_from_aws)

from . import s3

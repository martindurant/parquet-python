# encoding: utf-8
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

import datetime
from socket import timeout as socket_timeout

from kombu import Connection, Producer, Exchange
from pytz import timezone
from mozillapulse.utils import time_to_string

from mo_logs import constants
from pyLibrary import jsons
from mo_logs.exceptions import Except, suppress_exception
from mo_logs import Log
from mo_dots import wrap, coalesce, Data, set_default
from mo_kwargs import override
from mo_threads import Thread, Lock
from mozillapulse.consumers import GenericConsumer

count_locker=Lock()
count=0


class Consumer(Thread):
    @override
    def __init__(
        self,
        exchange,  # name of the Pulse exchange
        topic,  # message name pattern to subscribe to  ('#' is wildcard)
        target=None,  # WILL BE CALLED WITH PULSE PAYLOADS AND ack() IF COMPLETE$ED WITHOUT EXCEPTION
        target_queue=None,  # (aka self.queue) WILL BE FILLED WITH PULSE PAYLOADS
        host='pulse.mozilla.org',  # url to connect,
        port=5671,  # tcp port
        user=None,
        password=None,
        vhost="/",
        start=0,  # USED AS STARTING POINT FOR ASSIGNING THE _meta.count ATTRIBUTE
        ssl=True,
        applabel=None,
        heartbeat=False,  # True to also get the Pulse heartbeat message
        durable=False,  # True to keep queue after shutdown
        serializer='json',
        broker_timezone='GMT',
        kwargs=None
    ):
        global count
        count = coalesce(start, 0)

        self.target_queue = target_queue
        self.pulse_target = target
        if (target_queue == None and target == None) or (target_queue != None and target != None):
            Log.error("Expecting a queue (for fast digesters) or a target (for slow digesters)")

        Thread.__init__(self, name="Pulse consumer for " + kwargs.exchange, target=self._worker)
        self.settings = kwargs
        kwargs.callback = self._got_result
        kwargs.user = coalesce(kwargs.user, kwargs.username)
        kwargs.applabel = coalesce(kwargs.applable, kwargs.queue, kwargs.queue_name)
        kwargs.topic = topic

        self.pulse = ModifiedGenericConsumer(kwargs, connect=True, **kwargs)
        self.start()

    def _got_result(self, data, message):
        global count

        data = wrap(data)
        with count_locker:
            Log.note("{{count}} from {{exchange}}", count=count, exchange=self.pulse.exchange)
            data._meta.count = count
            data._meta.exchange = self.pulse.exchange
            count += 1

        if self.settings.debug:
            Log.note("{{data}}",  data= data)
        if self.target_queue != None:
            try:
                self.target_queue.add(data)
                message.ack()
            except Exception as e:
                e = Except.wrap(e)
                if not self.target_queue.closed:  # EXPECTED TO HAPPEN, THIS THREAD MAY HAVE BEEN AWAY FOR A WHILE
                    raise e
        else:
            try:
                self.pulse_target(data)
                message.ack()
            except Exception as e:
                Log.warning("Problem processing pulse (see `data` in structured log)", data=data, cause=e)

    def _worker(self, please_stop):
        def disconnect():
            with suppress_exception:
                self.target_queue.close()
                Log.note("stop put into queue")

            self.pulse.disconnect()
            Log.note("pulse listener was given a disconnect()")

        please_stop.on_go(disconnect)

        while not please_stop:
            try:
                self.pulse.listen()
            except Exception as e:
                if not please_stop:
                    Log.warning("Pulse had problem (Have you set your Pulse permissions correctly?", e)
        Log.note("pulse listener is done")


    def __exit__(self, exc_type, exc_val, exc_tb):
        Log.note("clean pulse exit")
        self.please_stop.go()
        with suppress_exception:
            self.target_queue.close()
            Log.note("stop put into queue")

        try:
            self.pulse.disconnect()
        except Exception as e:
            Log.warning("Can not disconnect during pulse exit, ignoring", e)
        Thread.__exit__(self, exc_type, exc_val, exc_tb)


class Publisher(object):
    """
    Mimic GenericPublisher https://github.com/bhearsum/mozillapulse/blob/master/mozillapulse/publishers.py
    """

    @override
    def __init__(
        self,
        exchange,  # name of the Pulse exchange
        host='pulse.mozilla.org',  # url to connect,
        port=5671,  # tcp port
        user=None,
        password=None,
        vhost="/",
        start=0,  # USED AS STARTING POINT FOR ASSIGNING THE _meta.count ATTRIBUTE
        ssl=True,
        applabel=None,
        heartbeat=False,  # True to also get the Pulse heartbeat message
        durable=False,  # True to keep queue after shutdown
        serializer='json',
        broker_timezone='GMT',
        kwargs=None
    ):
        self.settings = kwargs
        self.connection = None
        self.count = 0

    def connect(self):
        if not self.connection:
            self.connection = Connection(
                hostname=self.settings.host,
                port=self.settings.port,
                userid=self.settings.user,
                password=self.settings.password,
                virtual_host=self.settings.vhost,
                ssl=self.settings.ssl
            )

    def disconnect(self):
        if self.connection:
            self.connection.release()
            self.connection = None

    def send(self, topic, message):
        """Publishes a pulse message to the proper exchange."""

        if not message:
            Log.error("Expecting a message")

        message._prepare()

        if not self.connection:
            self.connect()

        producer = Producer(
            channel=self.connection,
            exchange=Exchange(self.settings.exchange, type='topic'),
            routing_key=topic
        )

        # The message is actually a simple envelope format with a payload and
        # some metadata.
        final_data = Data(
            payload=message.data,
            _meta=set_default({
                'exchange': self.settings.exchange,
                'routing_key': message.routing_key,
                'serializer': self.settings.serializer,
                'sent': time_to_string(datetime.datetime.now(timezone(self.settings.broker_timezone))),
                'count': self.count
            }, message.metadata)
        )

        producer.publish(jsons.scrub(final_data), serializer=self.settings.serializer)
        self.count += 1


class ModifiedGenericConsumer(GenericConsumer):
    def _drain_events_loop(self):
        while True:
            try:
                self.connection.drain_events(timeout=self.timeout)
            except socket_timeout, e:
                Log.warning("timeout! Restarting {{name}} pulse consumer.", name=self.exchange, cause=e)
                try:
                    self.disconnect()
                except Exception as f:
                    Log.warning("Problem with disconnect()", cause=f)
                break

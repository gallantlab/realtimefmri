import time
import serial
import struct
import argparse
import asyncio
import zmq
import zmq.asyncio
import evdev

from realtimefmri import utils
from realtimefmri import config


class Synchronize(object):
    """Detect and record pulses from the scanner. Can record to a local log
    file or transmit to a network destination.
    """
    def __init__(self, ttl_source='keyboard', verbose=True):
        logger = utils.get_logger('scanner', to_console=verbose, to_network=True)

        if ttl_source == 'keyboard':
            collect_ttl = self._collect_ttl_keyboard
        elif ttl_source == 'simulate':
            collect_ttl = self._collect_ttl_simulate
        elif ttl_source == 'serial':
            collect_ttl = self._collect_ttl_serial
        elif ttl_source == 'zmq':
            collect_ttl = self._collect_ttl_zmq
        else:
            raise NotImplementedError("TTL source {} not implemented.".format(ttl_source))
        logger.info(f'receiving TTL from {ttl_source}')

        context = zmq.Context()

        self.active = True
        self.context = context
        self.logger = logger
        self.collect_ttl = collect_ttl

    def collect_syncs(self):
        """Receive TTL pulses from scanner that indicate the start of volume
        acquisition and add them to a queue
        """
        socket = self.context.socket(zmq.PULL)
        socket.connect(config.SYNC_ADDRESS)

        while True:
            sync_time = yield from socket.recv()
            sync_time = struct.unpack('d', sync_time)[0]
            self.sync_queue.put(sync_time)

    @asyncio.coroutine
    def _collect_ttl_keyboard(self):
        keyboard = evdev.InputDevice(config.TTL_KEYBOARD_DEV)
        while self.active:
            try:
                events = keyboard.read()
                for event in events:
                    event = evdev.categorize(event)
                    if (isinstance(event, evdev.KeyEvent) and
                       (event.keycode == 'KEY_5') and  # 5 key
                       (event.keystate == event.key_down)):
                        yield time.time()
            except BlockingIOError:
                time.sleep(0.1)

    @asyncio.coroutine
    def _collect_ttl_serial(self):
        img_msg = 'TR'
        ser = serial.Serial(config.TTL_SERIAL_DEV)
        while self.active:
            msg = ser.read()
            if msg == img_msg:
                yield time.time()

    @asyncio.coroutine
    def _collect_ttl_zmq(self):
        socket = self.context.socket(zmq.PULL)
        socket.connect(config.TTL_ZMQ_ADDR)

        while self.active:
            _ = socket.recv()
            yield time.time()

    @asyncio.coroutine
    def _collect_ttl_simulate(self):
        while self.active:
            yield time.time()
            time.sleep(2)

    def run(self):
        socket = self.context.socket(zmq.PUSH)
        socket.bind(config.SYNC_ADDRESS)

        for t in self.collect_ttl():
            self.logger.info('TR %s', t)
            socket.send(struct.pack('d', t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', default='keyboard', action='store', type='str')
    args = parser.parse_args()
    sync = Synchronize(ttl_source=args.source)

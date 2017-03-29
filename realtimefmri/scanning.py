import sys
import time
import serial
import struct
import asyncio
import zmq
import zmq.asyncio
import evdev
from realtimefmri.utils import get_logger
from realtimefmri.config import SYNC_PORT, KEYBOARD_FN, TTL_PORT


class Scanner(object):
    '''Detect and record pulses from the scanner. Can record to a local log
    file or transmit to a network destination.
    '''
    def __init__(self, simulate=False, loop=None,
                 log_dest=['console', 'network']):
        to_console = True if 'console' in log_dest else False
        to_network = True if 'network' in log_dest else False
        logger = get_logger('scanner', to_console=to_console, to_network=to_network)

        if simulate:
            collect_function = self._simulate
        else:
            if is_available_port(TTL_PORT):
                print('using serial')
                collect_function = self._serial
            else:
                print('using keyboard')
                collect_function = self._keyboard

        context = zmq.asyncio.Context()

        if loop is None:
            loop = zmq.asyncio.ZMQEventLoop()

        asyncio.set_event_loop(loop)

        sync_queue = asyncio.Queue(loop=loop)

        self.active = True

        self.context = context
        self.loop = loop
        self.logger = logger
        self.collect_function = collect_function
        self.sync_queue = sync_queue

    @asyncio.coroutine
    def consume_sync_queue(self):
        socket = self.context.socket(zmq.PUSH)
        socket.bind('tcp://127.0.0.1:{}'.format(SYNC_PORT).encode())
        while self.active:
            recv_time = yield from self.sync_queue.get()
            yield from socket.send(struct.pack('d', recv_time))

    @asyncio.coroutine
    def _keyboard(self, ):
        devices = [evdev.InputDevice(dev) for dev in evdev.list_devices()]
        for dev in devices:
            print(dev.fn, dev.name)
        print('using', KEYBOARD_FN)
        devices = [evdev.InputDevice(dev) for dev in evdev.list_devices()]
        keyboard = evdev.InputDevice(KEYBOARD_FN)
        while self.active:
            events = yield from keyboard.async_read()
            for event in events:
                if ((event.value == 458786) and  # 5 key
                    (event.code == 4) and
                    (event.type == 4)):
                    self.logger.info('TR')
                    yield from self.sync_queue.put(time.time())

    @asyncio.coroutine
    def _serial(self):
        img_msg = 'TR'
        ser = serial.Serial(TTL_PORT)
        while self.active:
            msg = ser.read()
            print(msg)
            if msg==img_msg:
                self.logger.info('TR')
                yield from self.sync_queue.put(time.time())

    @asyncio.coroutine
    def _simulate(self):
        while self.active:
            self.logger.info('TR')
            yield from self.sync_queue.put(time.time())
            yield from asyncio.sleep(2)

    def run(self):
        tasks = asyncio.gather(self.collect_function(),
                               self.consume_sync_queue(),
                              )

        try:
            self.loop.run_until_complete(tasks)
        except KeyboardInterrupt:
            print('shutting down TTL monitor')
            tasks.cancel()
            self.loop.close()


def is_available_port(port):
    try:
        serial.Serial(port)
        available = True
    except serial.SerialException:
        available = False
    return available

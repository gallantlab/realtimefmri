import time
import serial
import struct
import asyncio
import zmq
import zmq.asyncio
import evdev

from realtimefmri.utils import get_logger
from realtimefmri.config import SYNC_PORT, KEYBOARD_FN, TTL_SERIAL_PORT
from realtimefmri.device_utils import list_devices


class Scanner(object):
    """Detect and record pulses from the scanner. Can record to a local log
    file or transmit to a network destination.
    """
    def __init__(self, simulate=False, loop=None, verbose=False):
        logger = get_logger('scanner', to_console=verbose, to_network=True)

        if simulate:
            collect_function = self._simulate
            logger.info('simulating TTL')
        else:
            if is_available_port(TTL_SERIAL_PORT):
                collect_function = self._serial
                logger.info('receiving TTL over serial')

            else:
                collect_function = self._keyboard
                logger.info('receiving TTL from keyboard')

        context = zmq.asyncio.Context()

        if loop is None:
            loop = asyncio.get_event_loop()

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
    def _keyboard(self):
        list_devices()
        keyboard = evdev.InputDevice(KEYBOARD_FN)
        while self.active:
            events = yield from keyboard.async_read()
            for event in events:
                event = evdev.categorize(event)
                if (isinstance(event, evdev.KeyEvent) and
                   (event.keycode == 'KEY_5') and  # 5 key
                   (event.keystate == event.key_down)):
                    recv_time = time.time()
                    self.logger.info('TR %s', recv_time)
                    yield from self.sync_queue.put(recv_time)

    @asyncio.coroutine
    def _serial(self):
        img_msg = 'TR'
        ser = serial.Serial(TTL_SERIAL_PORT)
        while self.active:
            msg = ser.read()
            print(msg)
            if msg == img_msg:
                recv_time = time.time()
                self.logger.info('TR %s', recv_time)
                yield from self.sync_queue.put(time.time())

    @asyncio.coroutine
    def _simulate(self):
        while self.active:
            recv_time = time.time()
            self.logger.info('TR %s' % recv_time)
            yield from self.sync_queue.put(time.time())
            yield from asyncio.sleep(2)

    def run(self):
        return asyncio.gather(self.collect_function(),
                              self.consume_sync_queue())


def is_available_port(port):
    try:
        serial.Serial(port)
        return True
    except serial.SerialException:
        return False

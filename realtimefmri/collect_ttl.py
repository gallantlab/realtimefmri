import time
import struct
import serial
import redis
import evdev

from realtimefmri import utils
from realtimefmri import config


class CollectTTL(object):
    """Detect and record pulses from the scanner. Can record to a local log
    file or transmit to a network destination.
    """
    def __init__(self, source='keyboard', verbose=True):
        logger = utils.get_logger('scanner', to_console=verbose, to_network=True)

        if source == 'keyboard':
            collect_ttl = self._collect_ttl_keyboard
        elif source == 'simulate':
            collect_ttl = self._collect_ttl_simulate
        elif source == 'serial':
            collect_ttl = self._collect_ttl_serial
        elif source == 'redis':
            collect_ttl = self._collect_ttl_redis
        else:
            raise NotImplementedError("TTL source {} not implemented.".format(source))
        logger.info(f'Receiving TTL from {source}')

        self.active = True
        self.verbose = verbose
        self.logger = logger
        self.collect_ttl = collect_ttl
        self.redis_client = redis.StrictRedis(host=config.REDIS_HOST)

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

    def _collect_ttl_simulate(self):
        while self.active:
            yield time.time()
            time.sleep(2)

    def _collect_ttl_serial(self):
        target_message = 'TR'
        ser = serial.Serial(config.TTL_SERIAL_DEV)
        while self.active:
            message = ser.read()
            if message == target_message:
                yield time.time()

    def _collect_ttl_redis(self):
        p = self.redis_client.pubsub()
        p.subscribe('ttl')
        for message in p.listen():
            if message['type'] == 'message':
                yield time.time()

    def collect(self):
        for t in self.collect_ttl():
            if self.verbose:
                self.logger.info('Received TTL at time {}'.format(t))
            self.redis_client.lpush('timestamp', struct.pack('d', t))


def collect_ttl(source, verbose=True):
    collector = CollectTTL(source, verbose=verbose)
    collector.collect()

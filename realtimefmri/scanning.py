import sys
import time
import serial
import struct
import zmq
import evdev
from realtimefmri.utils import get_logger
from realtimefmri.config import SYNC_PORT, KEYBOARD_FN, TTL_PORT




def is_available_port(port):
    try:
        ser = serial.Serial(port)
        available = True
    except serial.SerialException:
        available = False
    return available


class Scanner(object):
    def __init__(self, simulate=False, log_dest=['console', 'network']):
        '''Detect and record pulses from the scanner. Can record to a local log
        file or transmit to a network destination.
        '''
        to_console = True if 'console' in log_dest else False
        to_network = True if 'network' in log_dest else False
        logger = get_logger('scanner', to_console=to_console, to_network=to_network)

        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind('tcp://127.0.0.1:{}'.format(SYNC_PORT))

        self.logger = logger
        self.socket = socket
        self.simulate = simulate

    def _keyboard(self):
        devices = [evdev.InputDevice(dev) for dev in evdev.list_devices()]
        for dev in devices:
            print(dev.fn, dev.name)
        print('using', KEYBOARD_FN)
        devices = [evdev.InputDevice(dev) for dev in evdev.list_devices()]
        keyboard = evdev.InputDevice(KEYBOARD_FN)
        for event in keyboard.read_loop():
            if ((event.value == 458786) and  # 5 key
                (event.code == 4) and
                (event.type == 4)):
                self.logger.info('TR')
                self.socket.send(struct.pack('d', time.time()))

    def _serial(self):
        img_msg = 'TR'
        ser = serial.Serial(TTL_PORT)
        while True:
            msg = ser.read()
            print(msg)
            if msg==img_msg:
                self.logger.info('TR')
                recv_time = time.time()
                self.socket.send(struct.pack('d', recv_time))

    def _simulate(self):
        while True:
            self.logger.info('TR')
            sync_time = time.time()
            print(sync_time)
            self.socket.send(struct.pack('d', sync_time))
            time.sleep(2)

    def run(self):
        if self.simulate:
            self._simulate()
        else:
            if is_available_port(TTL_PORT):
                print('using serial')
                self._serial()
            else:
                print('using keyboard')
                self._keyboard()

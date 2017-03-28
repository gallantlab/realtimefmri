import sys
import time
import serial
import struct
import zmq
from realtimefmri.utils import get_logger

SYNC_PORT = 5556


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

    def _run(self):
        img_msg = 'TR'
        ser = serial.Serial('/dev/ttyUSB0')
        while True:
            msg = ser.read()
            print(msg)
            if msg==img_msg:
                self.logger.info('TR')
                self.socket.send(struct.pack('d', time.time()))

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
            self._run()
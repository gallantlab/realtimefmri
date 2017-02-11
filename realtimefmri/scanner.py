import sys
import time
import serial
from .utils import get_logger

class Scanner(object):
    def __init__(self, simulate=False, log_dest=['console', 'network']):
        '''Detect and record pulses from the scanner. Can record to a local log file
        or transmit to a network destination.
        log_path (str): logs to a file at that path
                        'file' logs to the module log directory
                        None does not log to a file
        log_port (int): a port to send log entries to
                        'default' logs to default tcp port
                        None does not log to a port
        simulate (bool): if true, simulate pulses

        '''
        to_console = True if 'console' in log_dest else False
        to_network = True if 'network' in log_dest else False
        logger = get_logger('scanner', to_console=to_console, to_network=to_network)

        if simulate:
            self.run = self._simulate
        else:
            self.run = self._run

        self.logger = logger

    def _run(self):
        ser = serial.Serial('/dev/ttyUSB0')
        while True:
            msg = ser.read()
            print msg
            if msg==img_msg:
                self.logger.info('TR')
    def _simulate(self):
        while True:
            self.logger.info('TR')
            time.sleep(2)
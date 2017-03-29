#!/usr/bin/env python

import sys
from os import makedirs
import os.path as op
import six
if six.PY2:
    import cPickle as pickle
elif six.PY3:
    import pickle
import logging
import logging.handlers
import socketserver
import struct

from realtimefmri.utils import get_logger
from realtimefmri.config import RECORDING_DIR, LOG_LEVEL


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unpickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unpickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """

    allow_reuse_address = 1

    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


def main(recording_id):
    if recording_id is not None:
        log_path = op.join(RECORDING_DIR, recording_id, 'recording.log')
        if not op.exists(op.dirname(log_path)):
            makedirs(op.dirname(log_path))
            print('making recording directory {}'.format(op.dirname(log_path)))
        print('saving log file to {}'.format(log_path))
    else:
        log_path = False
    
    _ = get_logger('root', to_console=True, to_file=log_path,
                   level=LOG_LEVEL)

    tcpserver = LogRecordSocketReceiver()
    
    try:
        print('starting logging...')
        tcpserver.serve_until_stopped()
    except KeyboardInterrupt:
        print('shutting down logging')
        sys.exit(0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        recording_id = sys.argv[1]
    else:
        recording_id = None
    
    main(recording_id)

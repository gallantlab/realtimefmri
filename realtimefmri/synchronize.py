"""Collect files and align them to sync pulses from scanner

or coroutine waits for pulses adds them to queue
and coroutine waits for images, pulls pulse from pulse queue
join them in order

"""
import struct
import asyncio
import zmq
import zmq.asyncio

from realtimefmri.config import (SYNC_PORT, VOLUME_PORT,
                                 PREPROC_PORT, STIM_PORT)
from realtimefmri.utils import get_logger



class Synchronizer(object):
    def __init__(self, loop=None, verbose=False):

        logger = get_logger('synchronizer',
                            to_console=verbose,
                            to_network=True)

        context = zmq.asyncio.Context()
        if loop is None:
            loop = zmq.asyncio.ZMQEventLoop()
        asyncio.set_event_loop(loop)

        sync_queue = asyncio.Queue(loop=loop)
        volume_queue = asyncio.Queue(loop=loop)
        sync_times = dict()

        self.logger = logger
        self.context = context
        self.loop = loop
        self.sync_queue = sync_queue
        self.volume_queue = volume_queue
        self.sync_times = sync_times        

    @asyncio.coroutine
    def collect_syncs(self):
        """Receive TTL pulses from scanner that indicate the start of volume
        acquisition and add them to a queue
        """
        socket = self.context.socket(zmq.PULL)
        socket.connect('tcp://127.0.0.1:{}'.format(SYNC_PORT))

        while True:
            sync_time = yield from socket.recv()
            sync_time = struct.unpack('d', sync_time)[0]
            yield from self.sync_queue.put(sync_time)


    @asyncio.coroutine
    def timestamp_volumes(self):
        """Pair each image volume with its corresponding acquisition pulse
        """
        socket = self.context.socket(zmq.SUB)
        socket.connect('tcp://127.0.0.1:{}'.format(VOLUME_PORT))
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        while True:
            (_, image_id, _) = yield from socket.recv_multipart()
            image_id = struct.unpack('i', image_id)[0]
            sync_time = yield from self.sync_queue.get()
            self.sync_times[image_id] = sync_time


    @asyncio.coroutine
    def publish_preprocessed(self):
        """Send timestamped preprocessed data to subscribers
        """
        in_socket = self.context.socket(zmq.SUB)
        in_socket.connect('tcp://127.0.0.1:{}'.format(PREPROC_PORT))
        in_socket.setsockopt(zmq.SUBSCRIBE, b'')

        out_socket = self.context.socket(zmq.PUB)
        out_socket.bind('tcp://127.0.0.1:{}'.format(STIM_PORT))

        while True:
            (topic, image_id, msg) = yield from in_socket.recv_multipart()
            image_id = struct.unpack('i', image_id)[0]
            sync_time = self.sync_times[image_id]
            yield from out_socket.send_multipart([topic,
                                                  struct.pack('d', sync_time),
                                                  msg])

    def run(self):
        return asyncio.gather(self.collect_syncs(),
                              self.timestamp_volumes(),
                              self.publish_preprocessed())



if __name__ == '__main__':
    synchronizer = Synchronizer()
    synchronizer.run()
"""Collect files and align them to sync pulses from scanner

or coroutine waits for pulses adds them to queue
and coroutine waits for images, pulls pulse from pulse queue
join them in order

"""
import sys
import struct
import asyncio
import zmq
import zmq.asyncio
from realtimefmri.config import (SYNC_PORT, VOLUME_PORT,
                                 PREPROC_PORT, STIM_PORT)


def main():

    context = zmq.asyncio.Context()
    loop = zmq.asyncio.ZMQEventLoop()
    asyncio.set_event_loop(loop)

    sync_queue = asyncio.Queue(loop=loop)
    volume_queue = asyncio.Queue(loop=loop)
    sync_times = dict()


    @asyncio.coroutine
    def consume_syncs(sync_queue):
        socket = context.socket(zmq.PULL)
        socket.connect('tcp://127.0.0.1:{}'.format(SYNC_PORT))

        while True:
            sync_time = yield from socket.recv()
            sync_time = struct.unpack('d', sync_time)[0]
            yield from sync_queue.put(sync_time)


    @asyncio.coroutine
    def consume_volumes(sync_queue, volume_queue):
        socket = context.socket(zmq.SUB)
        socket.connect('tcp://127.0.0.1:{}'.format(VOLUME_PORT))
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        while True:
            (_, raw_image_id, _) = yield from socket.recv_multipart()
            sync_time = yield from sync_queue.get()
            sync_times[raw_image_id] = sync_time


    @asyncio.coroutine
    def produce_stimuli():
        in_socket = context.socket(zmq.SUB)
        in_socket.connect('tcp://127.0.0.1:{}'.format(PREPROC_PORT))
        in_socket.setsockopt(zmq.SUBSCRIBE, b'')

        out_socket = context.socket(zmq.PUB)
        out_socket.bind('tcp://127.0.0.1:{}'.format(STIM_PORT))

        while True:
            (topic, raw_image_id, msg) = yield from in_socket.recv_multipart()
            sync_time = sync_times[raw_image_id]
            yield from out_socket.send_multipart([topic,
                                                  struct.pack('d', sync_time),
                                                  msg])


    tasks = asyncio.gather(consume_syncs(sync_queue),
                           consume_volumes(sync_queue, volume_queue),
                           produce_stimuli())


    try:
        loop.run_until_complete(tasks)
    except KeyboardInterrupt:
        print('shutting down synchronizer')
        tasks.cancel()
        loop.close()


if __name__ == '__main__':
    main()
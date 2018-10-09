#!/usr/bin/env python3
import os.path as op
import struct
import pickle
from uuid import uuid4
import asyncio
import zmq
import zmq.asyncio
import aionotify
import pydicom
from dicom2nifti import convert_siemens
from realtimefmri.utils import get_logger
from realtimefmri import config


class Collector(object):
    """Class to manage monitoring and loading from a directory
    """
    def __init__(self, loop=None, verbose=True):
        """Monitor a directory

        Arguments
        ---------
        directory: str
            Path that will be monitored for new files
        extension: file extension to detect
        loop : asyncio loop
        verbose: if True, log to console
        """

        logger = get_logger('collector', to_console=verbose, to_network=True)
        logger.info('data collector initialized')

        context = zmq.asyncio.Context()
        if loop is None:
            loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)

        volume_queue = asyncio.Queue(loop=loop)
        file_monitor = FileMonitor(config.SCANNER_DIR, volume_queue, extension='.dcm', loop=loop)

        self.file_monitor = file_monitor
        self.logger = logger
        self.context = context
        self.loop = loop
        self.active = True
        self.logger = logger
        self.image_number = 0
        self.context = context
        self.volume_queue = volume_queue
        self.sync_queue = asyncio.Queue(loop=loop)

    @asyncio.coroutine
    def collect_syncs(self):
        """Receive TTL pulses from scanner that indicate the start of volume
        acquisition and add them to a queue
        """
        socket = self.context.socket(zmq.PULL)
        socket.connect(config.SYNC_ADDRESS)

        while self.active:
            sync_time = yield from socket.recv()
            sync_time = struct.unpack('d', sync_time)[0]
            yield from self.sync_queue.put(sync_time)

    @asyncio.coroutine
    def publish_volumes(self):
        """Continuously monitor for incoming volumes, merge with TTL timestamps, and send to 
        preprocessor
        """
        volume_socket = self.context.socket(zmq.PUB)
        volume_socket.bind(config.VOLUME_ADDRESS)

        while self.active:
            new_volume_path = yield from self.volume_queue.get()
            self.logger.info('volume %s', new_volume_path)
            sync_time = yield from self.sync_queue.get()
            self.logger.info('collected at %d', sync_time)

            dcm = [pydicom.read_file(new_volume_path)]
            nii = convert_siemens.dicom_to_nifti(dcm, None)['NII']

            self.logger.debug('%s %s', op.basename(new_volume_path), str(nii.shape))
            image_number = struct.pack('i', self.image_number)
            yield from volume_socket.send_multipart([b'image', image_number,
                                                     pickle.dumps(nii)])
            self.image_number += 1

    def gather(self):
        return asyncio.gather(self.collect_syncs(),
                              self.publish_volumes(),
                              self.file_monitor.collect_volumes())

    def run(self):
        self.loop.run_until_complete(self.gather())


class FileMonitor(object):
    """Event monitor for new files

    Parameters
    ----------
    directory : str
        The directory to monitor for new files

    queue : str
        Asyncio queue to which new file paths are added

    extension : str
        Only respond to files with a given extension
    """
    def __init__(self, directory, queue, extension='.dcm', loop=None):
        if loop is None:
            loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)

        watcher = aionotify.Watcher()
        watcher.watch(alias=directory, path=directory,
                      flags=aionotify.Flags.CREATE | aionotify.Flags.CLOSE_WRITE)

        self.queue = queue
        self.extension = extension
        self.directory = directory
        self.loop = loop
        self.watcher = watcher

    @asyncio.coroutine
    def collect_volumes(self):
        yield from self.watcher.setup(self.loop)

        while True:
            event = yield from self.watcher.get_event()
            path = op.join(event.alias, event.name)
            if op.isdir(path):
                self.watcher.watch(alias=path, path=path,
                                   flags=aionotify.Flags.CREATE | aionotify.Flags.CLOSE_WRITE)

            elif aionotify.Flags.CLOSE_WRITE in aionotify.Flags.parse(event.flags):
                yield from self.queue.put(path)

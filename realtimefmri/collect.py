#!/usr/bin/env python3
'''Data collection code
'''
import os
import os.path as op
import struct
import pickle
import asyncio
import zmq
import zmq.asyncio
import pydicom
from dicom2nifti import convert_siemens
from realtimefmri.utils import get_logger
from realtimefmri.config import VOLUME_PORT


class Collector(object):
    '''Class to manage monitoring and loading from a directory
    '''
    def __init__(self, port=VOLUME_PORT, directory=None, parent_directory=None,
                 extension='.dcm', loop=None, verbose=False):
        '''Monitor a directory
        Args:
            port: port to publish images to
            directory: path that will be monitored for new files
            parent_directory: if True, indicates that `directory` should be
                              monitored for the first new directory and then
                              that directory should be monitored for new files
            extension: file extension to detect
            verbose: if True, log to console
        '''

        logger = get_logger('collector', to_console=verbose, to_network=True)
        logger.info('data collector initialized')

        context = zmq.asyncio.Context()
        if loop is None:
            loop = asyncio.get_event_loop()

        volume_queue = asyncio.Queue(loop=loop)

        self.port = port
        self.directory = directory
        self.parent_directory = parent_directory
        self.extension = extension
        self.active = False
        self.logger = logger
        self.image_number = 0
        self.context = context
        self.volume_queue = volume_queue

    @asyncio.coroutine
    def detect_child(self, directory):
        """Return the first new folder that is created in the given folder"""
        monitor = MonitorDirectory(directory, extension='/')
        while True:
            new_directories = monitor.get_new_paths()
            if len(new_directories) > 0:
                self.logger.debug('detected new folder %s', new_directories[0])
                return op.join(directory, new_directories.pop())

            yield from asyncio.sleep(0.1)

    @asyncio.coroutine
    def publish_volumes(self):
        """Consume the volume queue, load binary volume data, and publish data
        to subscribers
        """
        socket = self.context.socket(zmq.PUB)
        socket.bind('tcp://127.0.0.1:%d' % self.port)

        while True:
            image_path = yield from self.volume_queue.get()
            yield from asyncio.sleep(0.25)  # give time for file to close
            dcm = [pydicom.read_file(image_path)]
            nii = convert_siemens.dicom_to_nifti(dcm, None)['NII']

            self.logger.debug('%s %s', op.basename(image_path), str(nii.shape))
            image_number = struct.pack('i', self.image_number)
            yield from socket.send_multipart([b'image', image_number,
                                              pickle.dumps(nii)])
            self.image_number += 1

    @asyncio.coroutine
    def collect_volumes(self):
        """Continuously glob the monitor directory and add new files to the
        volume queue
        """
        if self.parent_directory:
            self.logger.debug('detecting next subfolder in %s',
                              self.parent_directory)
            child_dir = yield from self.detect_child(self.parent_directory)
            self.directory = child_dir

        monitor = MonitorDirectory(self.directory,
                                   extension=self.extension)

        self.active = True
        while self.active:
            new_image_paths = monitor.get_new_paths()
            monitor.update(new_image_paths)
            while len(new_image_paths) > 0:
                new_image_path = new_image_paths.pop()
                self.logger.info('volume %s', new_image_path)
                yield from self.volume_queue.put(op.join(self.directory,
                                                         new_image_path))
            yield from asyncio.sleep(0.1)

    def run(self):
        return asyncio.gather(self.collect_volumes(),
                              self.publish_volumes())


class MonitorDirectory(object):
    '''
    monitor the file contents of a directory
    Example usage:
        m = MonitorDirectory(dir_path)
        # add a file to that directory
        new_image_paths = m.get_new_paths()
        # use the new images
        # update image paths list to contain newly acquired images
        m.update(new_image_paths)
        # no images added
        new_image_paths = m.get_new_paths()
        len(new_image_paths)==0 # True
    '''
    def __init__(self, directory, pattern=None, extension='.dcm'):
        if extension == '/':
            self._is_valid = self._is_valid_directories
        else:
            self._is_valid = self._is_valid_files

        self.directory = directory
        self.extension = extension
        self.image_paths = self.get_directory_contents()

    def _is_valid_directories(self, val):
        return op.isdir(op.join(self.directory, val))

    def _is_valid_files(self, val):
        return val.endswith(self.extension)

    def get_directory_contents(self):
        '''
        returns entire contents of directory with extension
        '''
        try:
            new_paths = set([i for i in os.listdir(self.directory)
                             if self._is_valid(i)])
        except OSError:
            new_paths = set()

        return new_paths

    def get_new_paths(self):
        '''Gets entire contents of directory and returns paths that were not
           present since last update
        '''
        directory_contents = self.get_directory_contents()
        if len(directory_contents) > len(self.image_paths):
            new_image_paths = set(directory_contents) - self.image_paths
        else:
            new_image_paths = set()

        self.image_paths = directory_contents

        return list(new_image_paths)

    def update(self, new_image_paths):
        '''Adds paths to set of all image paths'''
        if len(new_image_paths) > 0:
            self.image_paths = self.image_paths.union(new_image_paths)

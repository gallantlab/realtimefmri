'''Data collection code
'''
import os
import os.path as op
import shutil
import time
from glob import glob
from itertools import cycle
from uuid import uuid4
import asyncio
import zmq
import zmq.asyncio

from realtimefmri.utils import get_logger, get_temporary_file_name
from realtimefmri.config import get_example_data_directory

VOLUME_PORT = 5557


class Simulator(object):
    '''Class to simulate from sample directory
    '''
    def __init__(self, simulate_directory=None, destination_directory=None,
                 parent_directory=False, interval=1, verbose=False):

        if not op.exists(destination_directory):
            os.makedirs(destination_directory)

        self.interval = interval
        self.simulate_directory = simulate_directory
        self.destination_directory = destination_directory
        self.parent_directory = parent_directory
        self.logger = get_logger('simulator', to_console=verbose,
                                 to_network=True)

    def run(self):
        '''run
        '''
        self.active = True
        if self.parent_directory:
            dirname = get_temporary_file_name(self.destination_directory)
            self.destination_directory = dirname
            self.logger.debug('making simulation directory %s', dirname)
            os.makedirs(dirname)

        image_fpaths = glob(op.join(self.simulate_directory, '*.PixelData'))
        image_fpaths.sort()
        self.logger.debug('simulating %u files from %s',
                         len(image_fpaths), self.simulate_directory)
        image_fpaths = cycle(image_fpaths)

        while self.active:
            image_fpath = next(image_fpaths)
            if self.interval == 'return':
                input('>> Press return for next image')
            else:
                time.sleep(self.interval)
            _, image_fname = op.split(image_fpath)
            new_image_fpath = op.join(self.destination_directory,
                                      str(uuid4())+'.PixelData')
            self.logger.info('copying %s to %s', image_fpath,
                             self.destination_directory)
            shutil.copy(image_fpath, new_image_fpath)
            time.sleep(0.2)  # simulate image scan and reconstruction time

    def stop(self):
        if self.parent_directory:
            root_directory, _ = op.split(self.destination_directory)
        else:
            root_directory = self.destination_directory

        self.logger.debug('removing %s', root_directory)
        shutil.rmtree(root_directory)

class Collector(object):
    '''Class to manage monitoring and loading from a directory
    '''
    def __init__(self, port=VOLUME_PORT, directory=None, parent_directory=None,
                 extension='.PixelData', loop=None, verbose=False):
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
            loop = zmq.asyncio.ZMQEventLoop
        asyncio.set_event_loop(loop)
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
                print('detected new folder {}'.format(new_directories[0]))
                return op.join(directory, new_directories.pop())

            yield from asyncio.sleep(0.1)

    @asyncio.coroutine
    def consume_volumes(self):
        """Consume the volume queue, load binary volume data, and publish data
        to subscribers
        """
        socket = self.context.socket(zmq.PUB)
        socket.bind('tcp://127.0.0.1:%d' % self.port)

        while True:
            image_fpath = yield from self.volume_queue.get()
            yield from asyncio.sleep(0.1)  # give time for file to close
            with open(image_fpath, 'rb') as f:
                raw_image_binary = f.read()

            self.logger.info('%s %u', op.basename(image_fpath),
                             len(raw_image_binary))

            image_number = '{:08}'.format(self.image_number).encode()
            yield from socket.send_multipart([b'image', image_number,
                                              raw_image_binary])

    @asyncio.coroutine
    def produce_volumes(self):
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
                if (self.image_number % 2) == 1: # only use odd/magnitude images
                    self.logger.debug('new image at %s', new_image_path)
                    yield from self.volume_queue.put(op.join(self.directory,
                                                             new_image_path))

                self.image_number += 1

            yield from asyncio.sleep(0.1)

    def run(self):
        return asyncio.gather(self.produce_volumes(),
                              self.consume_volumes())


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
    def __init__(self, directory, extension='.PixelData'):
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

        return list(new_image_paths)

    def update(self, new_image_paths):
        '''Adds paths to set of all image paths'''
        if len(new_image_paths) > 0:
            self.image_paths = self.image_paths.union(new_image_paths)

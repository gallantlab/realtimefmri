'''Data collection code
'''
import os
import os.path as op
import shutil
import time
from glob import glob
from itertools import cycle
import zmq

from realtimefmri.utils import get_logger, get_temporary_file_name
from realtimefmri.config import get_example_data_directory


class Simulator(object):
    '''Class to simulate from sample directory
    '''
    def __init__(self, out_port=5556, simulate_directory=None,
                 destination_directory=None, parent_directory=False,
                 interval=2, verbose=False):

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

        ex_dir = get_example_data_directory(self.simulate_directory)
        image_fpaths = glob(op.join(ex_dir, '*.PixelData'))
        image_fpaths.sort()
        self.logger.debug('simulating %u files from %s',
                         len(image_fpaths), ex_dir)
        image_fpaths = cycle(image_fpaths)

        while self.active:
            image_fpath = next(image_fpaths)
            if self.interval == 'return':
                raw_input('>> Press return for next image')
            else:
                time.sleep(self.interval)
            _, image_fname = op.split(image_fpath)
            new_image_fpath = op.join(self.destination_directory,
                                      image_fname)
            self.logger.debug('copying %s to %s', image_fpath,
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
    def __init__(self, out_port=5556, directory=None, parent_directory=False,
                 extension='.PixelData', verbose=False):
        '''Monitor a directory
        Args:
            out_port: port to publish images to
            directory: path that will be monitored for new files
            parent_directory: if True, indicates that `directory` should be
                              monitored for the first new directory and then
                              that directory should be monitored for new files
            extension: file extension to detect
            verbose: if True, log to console
        '''

        logger = get_logger('collecting', to_console=verbose, to_network=True)
        logger.info('data collector initialized')

        context = zmq.Context()
        self.image_pub = context.socket(zmq.PUB)
        self.image_pub.bind('tcp://*:%d' % out_port)
        self.parent_directory = parent_directory
        self.directory = directory
        self.extension = extension
        self.active = False
        self.logger = logger

    def send_image(self, image_fpath):
        '''Load image from path and publish to subscribers
        '''
        with open(image_fpath, 'r') as f:
            raw_image_binary = f.read()

        self.logger.info('%s %u', op.basename(image_fpath),
                         len(raw_image_binary))

        self.image_pub.send_multipart([b'image', raw_image_binary])

    def detect_parent(self):
        monitor = MonitorDirectory(self.directory, image_extension='/')
        while True:
            new_image_paths = monitor.get_new_image_paths()
            if len(new_image_paths) > 0:
                self.directory = op.join(self.directory,
                                              new_image_paths.pop())
                self.logger.debug('detected new folder %s monitoring',
                                 self.directory)
                break
            time.sleep(0.1)


    def run(self):
        '''Run continuously
        '''
        if self.parent_directory:
            self.logger.debug('detecting next subfolder in %s', self.directory)
            self.detect_parent()

        monitor = MonitorDirectory(self.directory,
                                   image_extension=self.extension)

        self.active = True
        while self.active:
            new_image_paths = monitor.get_new_image_paths()
            monitor.update(new_image_paths)
            while len(new_image_paths) > 0:
                new_image_path = new_image_paths.pop()
                self.logger.debug('new image at %s', new_image_path)
                self.send_image(op.join(self.directory, new_image_path))
            time.sleep(0.1)


class MonitorDirectory(object):
    '''
    monitor the file contents of a directory
    Example usage:
        m = MonitorDirectory(dir_path)
        # add a file to that directory
        new_image_paths = m.get_new_image_paths()
        # use the new images
        # update image paths list to contain newly acquired images
        m.update(new_image_paths)
        # no images added
        new_image_paths = m.get_new_image_paths()
        len(new_image_paths)==0 # True
    '''
    def __init__(self, directory, image_extension='.PixelData'):
        if image_extension == '/':
            self._is_valid = self._is_valid_directories
        else:
            self._is_valid = self._is_valid_files

        self.directory = directory
        self.image_extension = image_extension
        self.image_paths = self.get_directory_contents()

    def _is_valid_directories(self, val):
        return op.isdir(op.join(self.directory, val))

    def _is_valid_files(self, val):
        return val.endswith(self.image_extension)

    def get_directory_contents(self):
        '''
        returns entire contents of directory with image_extension
        '''
        try:
            new_paths = set([i for i in os.listdir(self.directory)
                             if self._is_valid(i)])
        except OSError:
            new_paths = set()
        
        return new_paths

    def get_new_image_paths(self):
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

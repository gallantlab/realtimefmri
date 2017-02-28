import os
import time
import struct
import functools
from glob import glob
import zmq
from itertools import cycle

from .utils import get_example_data_directory, get_logger

class DataCollector(object):
    def __init__(self, acq_port=5554, out_port=5556, directory=None, parent_directory=False, simulate=False, interval=None, verbose=False):
        '''
        Arguments
        directory, string with directory to watch for files, or if parent_directory is True, new folder
        parent_directory, bool indicating if provided directory is the image directory or a parent directory to watch
        simulate, bool, True indicates we should simulate from provided directory
        interval, return, or int indicating seconds between simulated image acquisition
        '''

        super(DataCollector, self).__init__()
        logger = get_logger('collecting', to_console=verbose, to_network=True) 
        logger.debug('data collector initialized')

        self.directory = directory
        self.parent_directory = parent_directory

        context = zmq.Context()
        self.acq_port = acq_port
        self.image_acq = context.socket(zmq.SUB)
        self.image_acq.connect('tcp://localhost:%d'%acq_port)
        self.image_acq.setsockopt(zmq.SUBSCRIBE, 'time')

        self.image_pub = context.socket(zmq.PUB)
        self.image_pub.bind('tcp://*:%d'%out_port)

        self.active = False
        self.logger = logger
        if simulate:
            self._run = functools.partial(self._simulate, interval=interval,
                                          directory=directory)

    def _simulate(self, interval, directory):
        ex_dir = get_example_data_directory(directory)
        image_fpaths = glob(os.path.join(ex_dir, '*.PixelData'))
        image_fpaths.sort()
        self.logger.info('simulating {} files from {}'.format(len(image_fpaths), ex_dir))
        image_fpaths = cycle(image_fpaths)

        for image_fpath in image_fpaths:
            if interval=='return':
                raw_input('>> Press return for next image')
            else:
                time.sleep(interval)
            time.sleep(0.2) # simulate image scan and reconstruction time

            with open(image_fpath, 'r') as f:
                raw_image_binary = f.read()
            self.logger.info('{} {}'.format(os.path.basename(image_fpath), len(raw_image_binary)))
            self.send_image(raw_image_binary)
            

    def send_image(self, raw_image_binary):
        self.image_pub.send_multipart([b'image', raw_image_binary])

    def _run(self):
        self.active = True
        self.monitor = MonitorDirectory(self.directory, image_extension='.PixelData')
        while self.active:
            new_image_paths = None
            while not new_image_paths:
                new_image_paths = self.monitor.get_new_image_paths()
                if len(new_image_paths)>0:
                    with open(os.path.join(self.directory, list(new_image_paths)[0]), 'r') as f:
                        raw_image_binary = f.read()
                    msg = 'image '+raw_image_binary
                    self.send_image(raw_image_binary)
                    self.monitor.update(new_image_paths)
                time.sleep(0.1)
    
    def run(self):
        # watch the parent_directory for the first new directory, then use that as the directory to monitor
        if self.parent_directory:
            m = MonitorDirectory(self.directory, image_extension='/')
            while True:
                new_image_paths = m.get_new_image_paths()
                if len(new_image_paths)>0:
                    self.directory = os.path.join(self.directory, new_image_paths.pop())
                    self.logger.info('detected new folder %s, monitoring' % self.directory)
                    break
                time.sleep(0.1)
        self._run()

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
        if image_extension=='/':
            self._is_valid = self._is_valid_directories
        else:
            self._is_valid = self._is_valid_files

        self.directory = directory
        self.image_extension = image_extension
        self.image_paths = self.get_directory_contents()

    def _is_valid_directories(self, val):
        return os.path.isdir(os.path.join(self.directory, val))
    def _is_valid_files(self, val):
        return val.endswith(self.image_extension)

    def get_directory_contents(self):
        '''
        returns entire contents of directory with image_extension
        '''
        return set([i for i in os.listdir(self.directory) if self._is_valid(i)])

    def get_new_image_paths(self):
        '''
        gets entire contents of directory and returns paths that were not present since last update
        '''
        directory_contents = self.get_directory_contents()
        if len(directory_contents)>len(self.image_paths):
            new_image_paths = set(directory_contents) - self.image_paths
        else: new_image_paths = set()

        return new_image_paths

    def update(self, new_image_paths):
        if len(new_image_paths)>0:
            self.image_paths = self.image_paths.union(new_image_paths)

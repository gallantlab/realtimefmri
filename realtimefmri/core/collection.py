import os
import time
import struct
import functools
from glob import glob
import zmq
from itertools import cycle

from .utils import get_example_data_directory, get_logger
logger = get_logger('collect.ion') 

class DataCollector(object):
	def __init__(self, out_port=5556, acq_port=5554, directory=None, parent_directory=False, simulate=None, interval=None):
		super(DataCollector, self).__init__()
		logger.debug('data collector initialized')

		self.directory = directory
		self.parent_directory = parent_directory

		context = zmq.Context()
		self.image_acq = context.socket(zmq.SUB)
		self.image_acq.connect('tcp://localhost:%d'%acq_port)
		self.image_acq.setsockopt(zmq.SUBSCRIBE, 'time')

		self.image_pub = context.socket(zmq.PUB)
		self.image_pub.bind('tcp://*:%d'%out_port)

		self._sync_with_subscriber(out_port+1)
		self._t0 = None

		self.active = False 

		if not simulate is None:
			self._run = functools.partial(self._simulate, interval=interval, subject=simulate)

	def _sync_with_subscriber(self, port):
		ctx = zmq.Context.instance()
		s = ctx.socket(zmq.REP)
		s.bind('tcp://*:%d'%port)
		logger.info('waiting for image subscriber to initialize sync')
        s.recv()
        s.send('READY!')
        logger.info('synchronized with image subscriber')

    def _sync_with_image_acq(self):
        logger.info('waiting for image')
        self.image_acq.recv()
        logger.info('acquired image')
        if self._t0 is None:
            self._t0 = time.time()
            logger.info('synchronized with first image at time %.2f'%self._t0)
        return time.time()-self._t0

    def _simulate(self, interval, subject):
        ex_dir = get_example_data_directory(subject)
        logger.info('simulating from %s' % ex_dir)
        image_fpaths = glob(os.path.join(ex_dir, '*.PixelData'))
        image_fpaths.sort()
        image_fpaths = cycle(image_fpaths)

        if interval=='sync':
            ctx = zmq.Context.instance()
            s = ctx.socket(zmq.PULL)
            s.connect('tcp://localhost:5554')

        for image_fpath in image_fpaths:
            if interval=='return':
                raw_input('>> Press return for next image')
            elif interval=='sync':
                t = self._sync_with_image_acq()
                time.sleep(0.2) # simulate image scan and reconstruction time
            else:
                time.sleep(interval)

            with open(image_fpath, 'r') as f:
                raw_image_binary = f.read()
            logger.info(os.path.basename(image_fpath))
            self.image_pub.send_multipart([b'image', struct.pack('d', t), raw_image_binary])

    def _run(self):
        self.active = True
        self.monitor = MonitorDirectory(self.directory, image_extension='.PixelData')
        while self.active:
                        new_image_paths = None
            t = self._sync_with_image_acq()
            while not new_image_paths:
                new_image_paths = self.monitor.get_new_image_paths()
                if len(new_image_paths)>0:
                    with open(os.path.join(self.directory, list(new_image_paths)[0]), 'r') as f:
                        raw_image_binary = f.read()
                    msg = 'image '+raw_image_binary
                    self.image_pub.send_multipart([b'image', t, raw_image_binary])
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
                    logger.info('detected new folder %s, monitoring' % self.directory)
                    break
                time.sleep(0.2)
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
        logger.debug('monitoring %s for %s' % (directory, image_extension))

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
            logger.debug(new_image_paths)
            self.image_paths = self.image_paths.union(new_image_paths)

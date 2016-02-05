import time
import logging
logger = logging.getLogger('collect.ion')
logger.setLevel(logging.DEBUG)

import functools

import os
import random
from glob import glob
import zmq

from itertools import cycle

from .utils import get_example_data_directory, get_log_directory

class DataCollector(object):
	def __init__(self, directory=None, parent_directory=False, simulate=None, interval=None):
		super(DataCollector, self).__init__()
		logger.debug('data collector initialized')

		self.directory = directory
		self.parent_directory = parent_directory

		context = zmq.Context()
		self.image_pub = context.socket(zmq.PUB)
		self.image_pub.bind('tcp://*:5556')
		self.active = False 
		if not simulate is None:
			self._run = functools.partial(self._simulate, interval=interval, subject=simulate)

	def _simulate(self, interval='return', subject='S1'):
		ex_dir = get_example_data_directory(subject)
		logger.info('simulating from %s' % ex_dir)
		image_fpaths = glob(os.path.join(ex_dir, '*.PixelData'))
		image_fpaths.sort()
		image_fpaths = cycle(image_fpaths)
		for image_fpath in image_fpaths:
			with open(image_fpath, 'r') as f:
				raw_image_binary = f.read()
			msg = 'image '+raw_image_binary
			logger.info('sending message of length %d\n(%s)' % (len(msg), os.path.basename(image_fpath)))
			self.image_pub.send(msg)
			if interval=='return':
				raw_input('>> Press return for next image')
			else:
				time.sleep(interval)

	def _run(self):
		self.active = True
		self.monitor = MonitorDirectory(self.directory, image_extension='.PixelData')
		while self.active:
			new_image_paths = self.monitor.get_new_image_paths()
			if len(new_image_paths)>0:
				with open(os.path.join(self.directory, list(new_image_paths)[0]), 'r') as f:
					raw_image_binary = f.read()
				msg = 'image '+raw_image_binary
				self.image_pub.send(msg)
			self.monitor.update(new_image_paths)
			time.sleep(0.2)
	
	def run(self):
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
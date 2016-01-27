import sys
import time
import logging
import argparse
import functools

import numpy as np
import time
import os
import random
from glob import glob
import zmq

from itertools import cycle

from utils import get_example_data_directory, get_log_directory

logger = logging.getLogger('data_collection')
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_log_directory(), '%s_data_collection.log'%time.strftime('%Y%m%d'))
formatter = logging.Formatter('%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s')
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

'''
actual data collection
-images arrive to scanner console from ICE
-images are stored in a shared folder on the local network
-real-time computer is sitting on that folder waiting for new images to appear
wait_for_image
-on new image, preprocesses image (register, detrend)
read_image
'''


class DataCollector(object):
	'''

	'''
	def __init__(self, directory, simulate=None, interval=None):
		super(DataCollector, self).__init__()
		logger.debug('data collector initialized')

		self.directory = directory

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
		logger.debug('monitoring %s'%directory)
		self.directory = directory
		self.image_extension = image_extension
		self.image_paths = self.get_directory_contents()

	def get_directory_contents(self):
		'''
		returns entire contents of directory with image_extension
		'''
		return set([i for i in os.listdir(self.directory) if i.endswith(self.image_extension)])

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

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Collect data')
	parser.add_argument('-s', '--simulate', 
		action='store', 
		dest='simulate', 
		default=None, 
		help='''Simulate data collection''')
	parser.add_argument('-i', '--interval',
		action='store',
		dest='interval',
		default='2',
		help='''Interval between scans, in seconds. Only active if simulate is True''')
	parser.add_argument('-d', '--directory',
		action='store',
		dest='directory',
		default='tmp',
		help='Directory to watch')
	args = parser.parse_args()

	try:
		interval = float(args.interval)
	except (TypeError, ValueError):
		if args.interval=='return':
			interval = args.interval
		else:
			raise ValueError

	d = DataCollector(args.directory, simulate=args.simulate, interval=interval)
	d.run()

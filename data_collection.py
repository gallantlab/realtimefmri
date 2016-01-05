import sys
import time
import logging
FORMAT = '%(processName)s %(process)d (%(levelname)s): %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
import numpy as np
import time
import os
import os.path as op
import random
from glob import iglob

import zmq

from itertools import cycle

'''
actual data collection
-images arrive to scanner console from ICE
-images are stored in a shared folder on the local network
-real-time computer is sitting on that folder waiting for new images to appear
wait_for_image
-on new image, preprocesses image (register, detrend)
read_image
'''

logging.basicConfig(level=logging.DEBUG)

class DataCollector(object):
	'''

	'''
	def __init__(self, directory, **kwargs):
		super(DataCollector, self).__init__()
		logging.debug('data collector initialized')

		context = zmq.Context()
		self.image_pub = context.socket(zmq.PUB)
		self.image_pub.bind('tcp://*:5556')
		self.active = False
		self.simulation = kwargs.get('simulate', False)
		simulate = kwargs.get('simulate', False)
		if simulate:
			self._run = self._simulate

	def _simulate(self, interval='return'):
		ex_dir = get_example_data_directory()
		logging.debug('simulating from %s' % ex_dir)
		image_fpaths = cycle(iglob(op.join(ex_dir, '*.PixelData')))
		for image_fpath in image_fpaths:
			with open(image_fpath, 'r') as f:
				dat = f.read()
			msg = 'image '+dat
			logging.debug('sending message of length %d' % len(msg))
			self.image_pub.send(msg)
			if interval=='return':
				raw_input('>> Press return for next image')
			else:
				time.sleep(interval)

	def _run(self):
		self.active = True
		self.monitor = MonitorDirectory(directory, image_extension='.PixelData')
		while self.active:
			new_image_paths = self.monitor.get_new_image_paths()
			if len(new_image_paths)>0:
				self.image_pub.send('image %s'%new_image_paths[0])
			self.monitor.update(new_image_paths)
			time.sleep(0.1)
	
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
		logging.debug('monitoring %s'%directory)
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
		for i in directory_contents:
			logging.debug('directory_contents %s' % directory_contents)
		if len(directory_contents)>len(self.image_paths):
			new_image_paths = set(directory_contents) - self.image_paths
		else: new_image_paths = set()

		return new_image_paths

	def update(self, new_image_paths):
		if len(new_image_paths)>0:
			logging.debug(new_image_paths)
			self.image_paths = self.image_paths.union(new_image_paths)

def get_example_data_directory():
	return '/Users/robert/Documents/gallant/example_data'

if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)
	d = DataCollector('tmp', simulate=True)
	d.run()

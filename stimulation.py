#!/usr/bin/env python
import os
import sys
import argparse

import time
import zmq

import threading

import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import yaml

import cortex

from utils import get_database_directory

db_dir = get_database_directory()

class Stimulator(object):
	def __init__(self, stim_config):
		
		with open(os.path.join(db_dir, stim_config+'.conf'), 'r') as f:
			self.pipeline = yaml.load(f)['pipeline']

		for step in self.pipeline:
			logger.debug('initializing %s' % step['name'])
			step['instance'].__init__(**step.get('kwargs', {}))

	def run(self):
		logger.debug('running')
		for step in self.pipeline:
			logger.debug('starting %s' % step['name'])
			step['instance'].start()


class Stimulus(threading.Thread):
	def __init__(self):
		super(Stimulus, self).__init__()

		self.daemon = True
		zmq_context = zmq.Context()
		self.input_socket = zmq_context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:5557')
		self.active = False

	@property
	def topic(self):
		return self._topic

	@topic.setter
	def topic(self, topic):
		self.input_socket.setsockopt(zmq.SUBSCRIBE, topic)
		self._topic = topic
		

	def run(self):
		self.active = True
		while self.active:
			logger.debug('waiting for message')
			msg = self.input_socket.recv()
			logger.debug('received message')
			self._run(msg)

class FlatMap(Stimulus):
	def __init__(self, topic, subject, xfm_name, mask_type, vmin=None, vmax=None):
		super(FlatMap, self).__init__()
		npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()
		
		data = np.zeros(npts)
		vol = cortex.Volume(data, subject, xfm_name)

		self.topic = topic

		self.subject = subject
		self.xfm_name = xfm_name
		self.mask_type = mask_type

		self.ctx_client = cortex.webshow(vol)
		self.vmin = vmin
		self.vmax = vmax

	def _run(self, msg):
		data = msg[len(self.topic)+1:]
		data = np.fromstring(data, dtype=np.float32)
		vol = cortex.Volume(data, self.subject, self.xfm_name, vmin=self.vmin, vmax=self.vmax)
		self.ctx_client.addData(data=vol)

class Debug(Stimulus):
	def __init__(self, topic):
		super(Debug, self).__init__()
		self.topic = topic

	def _run(self, msg):
		data = msg[len(self.topic)+1:]
		data = np.fromstring(data, dtype=np.float32)
		logger.debug(data)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Preprocess data')
	parser.add_argument('config',
		action='store',
		nargs='?',
		default='stim-01',
		help='Name of configuration file')
	args = parser.parse_args()

	stim = Stimulator(args.config)
	stim.run()
	try:
		while True:
			time.sleep(0.01)
	except KeyboardInterrupt:
		sys.exit(0)


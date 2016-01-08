#!/usr/bin/env python
import os
import sys

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
			self.stim_pipeline = yaml.load(f)['stimulation']

		for step in self.stim_pipeline:
			logger.debug('initializing %s' % step['name'])
			step['instance'].__init__()

	def run(self):
		logger.debug('running')
		for step in self.stim_pipeline:
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

	def _run(self):
		raise NotImplementedError

	def run(self):
		self.active = True
		while self.active:
			logger.debug('waiting for message')
			msg = self.input_socket.recv()
			logger.debug('received message')
			self._run(msg)

class FlatMap(Stimulus):
	def __init__(self):
		super(FlatMap, self).__init__()
		self.topic = 'gm_detrend'
		self.input_socket.setsockopt(zmq.SUBSCRIBE, self.topic)

	def _run(self, msg):
		data = msg[len(self.topic)+1:]
		data = np.fromstring(data, dtype=np.float32)
		logger.debug('FlatMap._run(); data length %i' % len(data))

if __name__=='__main__':
	stim = Stimulator('stim-01')
	stim.run()
	try:
		while True:
			time.sleep(0.01)
	except KeyboardInterrupt:
		sys.exit(0)


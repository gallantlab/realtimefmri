#!/usr/bin/env python
import os
import sys
import argparse

import time
import yaml
import zmq

import numpy as np
import logging

from core.utils import get_database_directory, get_log_directory
db_dir = get_database_directory()

# initialize root logger, assigning file handler to output messages to log file
logger = logging.getLogger('stimulation')
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_log_directory(), '%s_stimulation.log'%time.strftime('%Y%m%d'))
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class Stimulator(object):
	def __init__(self, stim_config):

		self.logger = logging.getLogger('stimulation.Stimulator')

		zmq_context = zmq.Context()
		self.input_socket = zmq_context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:5557')
		self.input_socket.setsockopt(zmq.SUBSCRIBE, '')
		self.active = False

		with open(os.path.join(db_dir, stim_config+'.conf'), 'r') as f:
			self.pipeline = yaml.load(f)['pipeline']

		for step in self.pipeline:
			self.logger.debug('initializing %s' % step['name'])
			step['instance'].__init__(**step.get('kwargs', {}))

	def run(self):
		self.active = True
		self.logger.info('running')
		while self.active:
			self.logger.debug('start receive wait')
			msg = self.input_socket.recv()
			self.logger.debug('received message')
			topic_end = msg.find(' ')
			topic = msg[:topic_end]
			data = msg[topic_end+1:]
			for stim in self.pipeline:
				if stim['topic']==topic:
					self.logger.info('sending data of length %i to %s'%(len(data), topic))
					stim['instance'].run(data)
					self.logger.info('%s function returned'%stim['name'])

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


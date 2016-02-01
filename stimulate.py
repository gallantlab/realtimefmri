#!/usr/bin/env python
import os
import sys
import argparse

import time
import yaml
import zmq

import numpy as np
import warnings

from core.utils import get_database_directory, get_log_directory, get_recording_directory, get_configuration_directory
config_dir = get_configuration_directory()
db_dir = get_database_directory()
rec_dir = get_recording_directory()

import logging
logger = logging.getLogger('stimulate')
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_log_directory(), '%s_stimulate.log'%time.strftime('%Y%m%d'))
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-20s %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Stimulator(object):
	def __init__(self, stim_config):

		self.logger = logging.getLogger('stimulation.Stimulator')

		zmq_context = zmq.Context()
		self.input_socket = zmq_context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:5557')
		self.input_socket.setsockopt(zmq.SUBSCRIBE, '')
		self.active = False

		with open(os.path.join(config_dir, stim_config+'.conf'), 'r') as f:
			config = yaml.load(f)
			self.pipeline = config['pipeline']
			self.global_defaults = config.get('global_defaults', dict())
		
		if self.global_defaults['record']:
			if self.global_defaults['recording_id'] is None:
				self.global_defaults['recording_id'] = '%s_%s'%(self.global_defaults['subject'], time.strftime('%Y%m%d_%H%M'))
			try:
				os.mkdir(os.path.join(rec_dir, self.global_defaults['recording_id']))
			except OSError:
				warnings.warn('Recording id %s already exists!' % self.global_defaults['recording_id'])

		for step in self.pipeline:
			self.logger.debug('initializing %s' % step['name'])

			params = step.get('kwargs', {})
			for k,v in self.global_defaults.iteritems():
				params.setdefault(k, v)

			print params

			step['instance'].__init__(**params)

	def run(self):
		self.active = True
		self.logger.info('running')
		while self.active:
			try:
				self.logger.debug('start receive wait')
				msg = self.input_socket.recv()
				self.logger.debug('received message')
				topic_end = msg.find(' ')
				topic = msg[:topic_end]
				data = msg[topic_end+1:]
				for stim in self.pipeline:
					if topic in stim['topic'].keys():
						self.logger.info(topic)
						self.logger.info(stim['topic'])
						self.logger.info('sending data of length %i to %s'%(len(data), topic))

						stim['instance'].run({stim['topic'][topic]: data})
						self.logger.info('%s function returned'%stim['name'])
			except (KeyboardInterrupt, SystemExit):
				self.active = False
				for stim in self.pipeline:
					stim['instance'].stop()

				sys.exit(0)


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Preprocess data')
	parser.add_argument('config',
		action='store',
		nargs='?',
		default='stim-01',
		help='Name of configuration file')
	args = parser.parse_args()

	stim = Stimulator(args.config)
	stim.run() # this will start an infinite run loop
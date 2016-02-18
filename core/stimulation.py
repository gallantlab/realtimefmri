import os
import sys
import time
import yaml
import warnings

import numpy as np
import json
import zmq
import cortex

from matplotlib import pyplot as plt
plt.ion()

from .utils import database_directory, recording_directory, configuration_directory, generate_command, get_logger
config_dir = configuration_directory
db_dir = database_directory
rec_dir = recording_directory

class Stimulator(object):
	def __init__(self, stim_config, in_port=5558):
		zmq_context = zmq.Context()
		self.input_socket = zmq_context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:%d'%in_port)
		self.input_socket.setsockopt(zmq.SUBSCRIBE, '')
		self.active = False

		with open(os.path.join(config_dir, stim_config+'.conf'), 'r') as f:
			config = yaml.load(f)
			self.initialization = config.get('initialization', dict())
			self.pipeline = config['pipeline']
			self.global_defaults = config.get('global_defaults', dict())
		
		if self.global_defaults['recording_id'] is None:
			self.global_defaults['recording_id'] = '%s_%s'%(self.global_defaults['subject'],
				time.strftime('%Y%m%d_%H%M'))
		try:
			self.rec_dir = os.path.join(rec_dir, self.global_defaults['recording_id'])
			os.makedirs(os.path.join(self.rec_dir, 'logs'))
		except OSError:
			warnings.warn('Recording id %s already exists!' % self.global_defaults['recording_id'])

		self.logger = get_logger('stimulate.ion', dest=[os.path.join(self.rec_dir, 'logs', 'stimulation.log')])
		self.logger.info('making recording directory for id %s' % self.global_defaults['recording_id'])

		for init in self.initialization:
			self.logger.debug('initializing %s' % init['name'])
			params = init.get('kwargs', {})	
			for k,v in self.global_defaults.iteritems():
				params.setdefault(k, v)
			init['instance'].__init__(**params)
		for step in self.pipeline:
			self.logger.debug('initializing %s' % step['name'])
			params = step.get('kwargs', {})
			for k,v in self.global_defaults.iteritems():
				params.setdefault(k, v)
			step['instance'].__init__(**params)

		self._sync_with_publisher(in_port+1)

	def _sync_with_publisher(self, port):
		ctx = zmq.Context.instance()
		s = ctx.socket(zmq.REQ)
		s.connect('tcp://localhost:%d'%port)
		self.logger.info('requesting synchronization with preprocessing publisher')
		s.send('READY?')
		self.logger.info('waiting for preprocessing publisher to respond to sync request')
		s.recv()
		self.logger.info('synchronized with preprocessing publisher')

	def run(self):
		self.active = True
		self.logger.info('running')
		for init in self.initialization:
			self.logger.debug('starting %s' % init['name'])
			init['instance'].run()

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
						self.logger.debug('sending data of length %i to %s'%(len(data), topic))
						stim['instance'].run({stim['topic'][topic]: data})
						self.logger.debug('%s function returned'%stim['name'])
			except (KeyboardInterrupt, SystemExit):
				self.active = False
				for init in self.initialization:
					init['instance'].stop()
				for stim in self.pipeline:
					stim['instance'].stop()
				sys.exit(0)

class Stimulus(object):
	def stop(self):
		pass
	def run(self):
		raise NotImplementedError

class PyCortexViewer(Stimulus):
	def __init__(self, subject, xfm_name, mask_type='thick', vmin=-1., vmax=1., **kwargs):
		super(PyCortexViewer, self).__init__()
		npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()
		
		data = np.zeros(npts)
		vol = cortex.Volume(data, subject, xfm_name)

		self.subject = subject
		self.xfm_name = xfm_name
		self.mask_type = mask_type

		self.ctx_client = cortex.webshow(vol)
		self.vmin = vmin
		self.vmax = vmax
		self.active = True

	def run(self, inp):
		if self.active:
			try:
				data = np.fromstring(inp['data'], dtype=np.float32)
				vol = cortex.Volume(data, self.subject, self.xfm_name, vmin=self.vmin, vmax=self.vmax)
				self.ctx_client.addData(data=vol)
			except IndexError:
				self.active = False

class ConsolePlot(Stimulus):
	def __init__(self, xmin=-2., xmax=2., width=40, **kwargs):
		super(ConsolePlot, self).__init__()
		self.xmin = xmin
		self.xmax = xmax
		self.x_range = xmax-xmin
		self.width = width
		self.y_range = width
	
	def make_bars(self, x):
		y = ((x-self.xmin)/self.x_range)*self.y_range
		middle = self.width/2
		y = min(y,self.width)
		y = max(y,0)
		if y<middle:
			left_space = [' ']*int(y)
			bar = ['-']*int(middle-y)
			right_space = [' ']*int(middle)
		elif y>middle:
			left_space = [' ']*int(middle)
			bar = ['-']*int(y-middle)
			right_space = [' ']*int(self.width-y)
		else:
			left_space = [' ']*int(middle)
			bar = ['|']
			right_space = [' ']*int(middle)
		return ''.join(left_space+bar+right_space)

	def run(self, inp):
		x = np.fromstring(inp['data'], dtype=np.float32)
		print self.make_bars(x)

class RoiBars(Stimulus):
	def __init__(self, **kwargs):
		super(RoiBars, self).__init__()
		self.fig = plt.figure();
		self.ax = self.fig.add_subplot(111);
		self.rects = None
		plt.show()
		plt.draw()
	def run(self, data):
		data = json.loads(data)
		if self.rects is None:

			self.rects = self.ax.bar(range(len(data)), data.values())
			plt.show()
			plt.draw()
		else:
			for r, v in zip(self.rects, data.values()):
				r.set_height(v)
			plt.show()
			plt.draw() # should update

class Debug(Stimulus):
	def run(self, data):
		data = np.fromstring(data, dtype=np.float32)
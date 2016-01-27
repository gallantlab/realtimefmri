import numpy as np
import zmq
import threading
import cortex

import logging
print 'stimuli printing name ',__name__
logger = logging.getLogger('stimulation.stimuli')
logger.setLevel(logging.DEBUG)

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
		self.logger = logging.getLogger('stimulation.stimuli.FlatMap')
		self.logger.debug('initialized FlatMap')

	def _run(self, msg):
		data = msg[len(self.topic)+1:]
		data = np.fromstring(data, dtype=np.float32)
		vol = cortex.Volume(data, self.subject, self.xfm_name, vmin=self.vmin, vmax=self.vmax)
		self.ctx_client.addData(data=vol)

class Debug(Stimulus):
	def __init__(self, topic):
		super(Debug, self).__init__()
		self.topic = topic
		self.logger = logging.getLogger('stimulation.stimuli.Debug')
		self.logger.debug('initialized Debug')

	def _run(self, msg):
		data = msg[len(self.topic)+1:]
		data = np.fromstring(data, dtype=np.float32)
		self.logger.info(data)
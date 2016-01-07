#!/usr/bin/env python
import time
import zmq
import cortex
import logging
logger = logging.getLogger(__name__)

class Stimulation(object):
	def __init__(self):
		zmq_context = zmq.Context()
		self.input_socket = zmq_context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:5557')

		self.active = False

	def run(self):
		self.active = True
		logger.debug('running')
		while self.active:
			message = self.input_socket.recv()
			data = message.strip('')
			time.sleep(0.1)



if __name__=='__main__':
	stim = Stimulation()
	stim.run()
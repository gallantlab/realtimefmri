import numpy as np
import json
import zmq
import threading
import multiprocessing
import cortex

import logging
logger = logging.getLogger('stimulation.stimuli')
logger.setLevel(logging.DEBUG)

import pyo

from matplotlib import pyplot as plt
plt.ion()

class FlatMap(object):
	def __init__(self, subject, xfm_name, mask_type, vmin=None, vmax=None):
		npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()
		
		data = np.zeros(npts)
		vol = cortex.Volume(data, subject, xfm_name)

		self.subject = subject
		self.xfm_name = xfm_name
		self.mask_type = mask_type

		self.ctx_client = cortex.webshow(vol)
		self.vmin = vmin
		self.vmax = vmax
		self.logger = logging.getLogger('stimulation.stimuli.FlatMap')
		self.logger.debug('initialized FlatMap')

	def run(self, data):
		logger.info(len(data))
		data = np.fromstring(data, dtype=np.float32)
		vol = cortex.Volume(data, self.subject, self.xfm_name, vmin=self.vmin, vmax=self.vmax)
		self.ctx_client.addData(data=vol)

class RoiBars(object):
	def __init__(self):
		self.fig, self.ax = plt.subplots();
		self.rects = None
	def run(self, data):
		data = json.loads(data)
		if self.rects is None:
			self.rects = self.ax.bar(range(len(data)), data.values())
			plt.show(block=False)
		else:
			logger.info(data.values())
			for r, v in zip(self.rects, data.values()):
				r.set_height(v)
			plt.draw() # should update

class WeirdSound(object):
	def __init__(self):
		self.server = pyo.Server().boot()
		self.server.start()

		self.lfo_freq0 = 0.4
		self.lfo_freq = pyo.SigTo(value=[self.lfo_freq0,
			self.lfo_freq0+(0.01*self.lfo_freq0)],
			time=0.5)
		self.lfo = pyo.LFO(freq=self.lfo_freq, mul=0.005)
		self.lfo.play()
		
		self.f0 = 180
		self.freq = pyo.SigTo(value=[self.f0, self.f0+(0.01*self.f0),
			self.f0*2, self.f0*2+(0.01*self.f0*2)],
			time=1.)

		self.synth = pyo.Sine(freq=self.freq, mul=self.lfo)
		self.synth.out()
	def run(self, controls):
		controls = json.loads(controls)
		f = self.f0*(1.+controls['M1H']*2.)
		self.freq.value = [f, f+(0.01*f), f*2, f*2+(0.01*f*2)]

		f = self.lfo_freq0*(1.+controls['M1F']*5.)
		self.lfo_freq.value = [f, f+(0.01*f)]

class Debug(object):
	def __init__(self):
		self.logger = logging.getLogger('stimulation.stimuli.Debug')
		self.logger.debug('initialized Debug')

	def run(self, data):
		data = np.fromstring(data, dtype=np.float32)
		self.logger.info(data)
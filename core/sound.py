import os
import subprocess
import numpy as np
import pyo64 as pyo
import logging
logger = logging.getLogger('stimulate.ion')

from .stimulation import Stimulus
from .utils import generate_command, get_recording_directory

rec_dir = get_recording_directory()
server = pyo.Server(audio='jack').boot()
server.start()

class WeirdSound2(Stimulus):
	def __init__(self, record=False, recording_id=None, **kwargs):
		super(WeirdSound2, self).__init__()

		freq = pyo.midiToHz(60)	
		self._freq = pyo.SigTo(value=[freq, freq+(0.01*freq),
			freq*2, freq*2+(0.01*freq*2)],
			time=1.)

		possible_notes = [pyo.midiToHz(i*12+j) for i in range(4,7) for j in [0,2,4,5,7,9,11]]
		self._env_table = pyo.CosTable([(0,0), (50,1), (250,.3), (8191,0)])
		# self.onset_generator = pyo.Euclide().play()
		self.onset_generator = pyo.Cloud(density=40, poly=10).play()
		self.note_generator = pyo.TrigChoice(self.onset_generator, possible_notes)
		self.env = pyo.TrigEnv(self.onset_generator, table=self._env_table, dur=0.25, mul=0.005)

		self.synth = pyo.Sine(freq=self.note_generator, mul=self.env).play()
		self._pan = pyo.SigTo(value=0.5, time=1.9)
		self.panner = pyo.Pan(self.synth, outs=2, pan=self._pan)
		self.panner.out()

		self.record = record
		if self.record:
			self.rec_path = os.path.join(rec_dir, recording_id, 'weirdsound.wav')
			try:
				os.makedirs(os.path.dirname(self.rec_path))
			except OSError:
				pass
			self.server.recstart(self.rec_path)

	@property
	def freq(self):
		return self._freq.value
	@freq.setter
	def freq(self, val):
		self._freq.value = [val, val+(0.01*val), val*2., val*2.+(0.01*val*2.)]
	
	@property
	def pan(self):
		return self._pan.value
	@pan.setter
	def pan(self, val):
		self._pan.value = val

	@property
	def note(self):
		return self._note
	@note.setter
	def note(self, val):
		self.freq = pyo.midiToHz(val)

	def _parse_input(self, inp):
		return {k:np.fromstring(v, dtype=np.float32) for (k,v) in inp.iteritems()}
	def _validate_input(self, inp):
		def is_valid_input(x):
			return not np.isnan(x)
		return {k:v for (k,v) in inp.iteritems() if is_valid_input(v)}

	def run(self, inp):
		inp = self._parse_input(inp)
		inp = self._validate_input(inp)
		if 'pan' in inp:
			self.pan = float(inp['pan'])
		if 'freq' in inp:
			f = self.f_initial*(1.+cv1*2.)
			self.freq = [f, f+(0.01*f), f*2, f*2+(0.01*f*2)]

	def stop(self):
		logger.info('stopping weird sound')
		self.synth.stop()
		self.onset_generator.stop()
		if self.record:
			self.server.recstop()
			params = [
				{'name': 'input', 'position': 'first', 'value': self.rec_path},
				{'name': 'output', 'position': 'last', 'value': self.rec_path.replace('.wav', '.mp3')}
			]
			cmd = generate_command('sox', params)
			subprocess.call(cmd)
			os.remove(self.rec_path)

class WeirdSound(Stimulus):
	def __init__(self, record=False, recording_id=None, **kwargs):
		super(WeirdSound, self).__init__()
		self._lfo_freq_initial = 0.4
		self._lfo_freq = pyo.SigTo(value=self._lfo_freq_initial, time=0.5)

		self.lfo = pyo.Sine(freq=self._lfo_freq, mul=0.001, add=0.001)
		self.lfo.play()
		
		freq = pyo.midiToHz(60)	
		self._freq = pyo.SigTo(value=[freq, freq+(0.01*freq),
			freq*2, freq*2+(0.01*freq*2)],
			time=1.)

		self.synth = pyo.Sine(freq=self._freq, mul=self.lfo)
		self._pan = pyo.SigTo(value=0.5, time=1.9)
		self.panner = pyo.Pan(self.synth, outs=2, pan=self._pan)
		self.panner.out()

		self.record = record
		if self.record:
			self.rec_path = os.path.join(rec_dir, recording_id, 'weirdsound.wav')
			try:
				os.makedirs(os.path.dirname(self.rec_path))
			except OSError:
				pass
			self.server.recstart(self.rec_path)

	@property
	def lfo_freq(self):
		return self._lfo_freq.value
	@lfo_freq.setter
	def lfo_freq(self, val):
		self._lfo_freq.value = val

	@property
	def freq(self):
		return self._freq.value
	@freq.setter
	def freq(self, val):
		self._freq.value = [val, val+(0.01*val), val*2., val*2.+(0.01*val*2.)]
	
	@property
	def pan(self):
		return self._pan.value
	@pan.setter
	def pan(self, val):
		self._pan.value = val

	@property
	def note(self):
		return self._note
	@note.setter
	def note(self, val):
		self.freq = pyo.midiToHz(val)

	def _parse_input(self, inp):
		return {k:np.fromstring(v, dtype=np.float32) for (k,v) in inp.iteritems()}
	def _validate_input(self, inp):
		def is_valid_input(x):
			return not np.isnan(x)
		return {k:v for (k,v) in inp.iteritems() if is_valid_input(v)}

	def run(self, inp):
		inp = self._parse_input(inp)
		inp = self._validate_input(inp)
		if 'pan' in inp:
			self.pan = float(inp['pan'])
		if 'freq' in inp:
			f = self.f_initial*(1.+cv1*2.)
			self.freq = [f, f+(0.01*f), f*2, f*2+(0.01*f*2)]
		if 'lfo_freq' in inp:
			f = self._lfo_freq_initial*(1.+cv2*5.)
			self.lfo_freq = [f, f+(0.01*f)]

	def stop(self):
		logger.info('stopping weird sound')
		self.synth.stop()
		if self.record:
			self.server.recstop()
			params = [
				{'name': 'input', 'position': 'first', 'value': self.rec_path},
				{'name': 'output', 'position': 'last', 'value': self.rec_path.replace('.wav', '.mp3')}
			]
			cmd = generate_command('sox', params)
			subprocess.call(cmd)
			os.remove(self.rec_path)

class AudioRecorder(Stimulus):
	def __init__(self, jack_port, file_name, recording_id, **kwargs):
		super(AudioRecorder, self).__init__()
		rec_path = os.path.join(rec_dir, recording_id, file_name+'.wav')
		try:
			os.makedirs(os.path.dirname(rec_path))
		except OSError:
			pass
		params = [
			{'name': 'out_path', 'flag': 'f', 'value': rec_path},
			{'name': 'duration', 'flag': 'd', 'value': '-1'},
			{'name': 'jack_port', 'position': 'last', 'value': jack_port}
		]
		self.cmd = generate_command('jack_rec', params)
		self.rec_path = rec_path
	def run(self):
		self.proc = subprocess.Popen(self.cmd)
	def stop(self):
		logger.debug('stopping AudioRecorder')
		self.proc.terminate()
		params = [
			{'name': 'input', 'position': 'first', 'value': self.rec_path},
			{'name': 'output', 'position': 'last', 'value': self.rec_path.replace('.wav', '.mp3')}
		]
		cmd = generate_command('sox', params)
		subprocess.call(cmd)
		os.remove(self.rec_path)
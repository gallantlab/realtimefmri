import os
import subprocess
import numpy as np
import pyo64 as pyo

from .stimulation import Stimulus
from .utils import generate_command, recording_directory, get_logger
logger = get_logger('stimulate.ion')

rec_dir = recording_directory
server = pyo.Server(audio='jack').boot()
server.start()

class SoundStimulus(Stimulus):
	def __init__(self, **kwargs):
		super(SoundStimulus, self).__init__()
		recording_id = kwargs.get('recording_id')
		self.record = kwargs.get('record')
		self.name = self.__class__.__name__.split('.')[-1]
		if self.record:
			self.rec_path = os.path.join(rec_dir,
				recording_id, 'logs',
				self.name+'.wav')

	def start(self):
		if self.record:
			server.recstart(self.rec_path)

	def stop(self):
		if self.record:
			server.recstop()
			params = [
				{'name': 'input', 'position': 'first', 'value': self.rec_path},
				{'name': 'output', 'position': 'last', 'value': self.rec_path.replace('.wav', '.mp3')}
			]
			cmd = generate_command('lame', params)
			with open(os.devnull, 'w') as devnull:
				subprocess.call(cmd, stdout=devnull, stderr=devnull)
			os.remove(self.rec_path)

class BrainJukebox(Stimulus):
	def __init__(self, sound_paths, **kwargs):
		super(BrainJukebox, self).__init__(**kwargs)
		self.nsounds = len(sound_paths)

class WeirdSound(SoundStimulus):
	def __init__(self, **kwargs):
		super(WeirdSound, self).__init__(**kwargs)
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

	def start(self):
		super(WeirdSound, self).start()
		self.panner.out()

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
		super(WeirdSound, self).stop()
		self.synth.stop()
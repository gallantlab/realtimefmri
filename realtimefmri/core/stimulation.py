import os
import sys
import time
import subprocess
import yaml
import warnings
import pdb

import numpy as np
import json
import zmq
import cortex

from matplotlib import pyplot as plt
plt.ion()

from .utils import database_directory, recording_directory, configuration_directory, generate_command, get_logger, NetworkTimedObject
config_dir = configuration_directory
db_dir = database_directory
rec_dir = recording_directory

class Stimulator(NetworkTimedObject):
    def __init__(self, stim_config, in_port=5558):
        super(Stimulator, self).__init__()
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

        self.logger = get_logger('stimulate.ion', dest=['console', os.path.join(self.rec_dir, 'logs', 'stimulation.log')])
        self.log('making recording directory for id %s' % self.global_defaults['recording_id'])

        for init in self.initialization:
            self.log('initializing %s' % init['name'])
            params = init.get('kwargs', {})    
            for k,v in self.global_defaults.iteritems():
                params.setdefault(k, v)
            init['instance'].__init__(**params)

        for step in self.pipeline:
            self.log('initializing %s' % step['name'])
            params = step.get('kwargs', {})
            for k,v in self.global_defaults.iteritems():
                params.setdefault(k, v)
            step['instance'].__init__(**params)

    def log(self, msg):
        self.logger.debug('log: {:<50s}{:12.4f}'.format(msg, self.timestamp))
    
    def run(self):
        self.active = True
        self.log('running')
        for init in self.initialization:
            self.log('starting %s' % init['name'])
            init['instance'].start()
            self.log('started %s' % init['name'])

        for stim in self.pipeline:
            self.log('starting %s' % stim['name'])
            stim['instance'].start()
            self.log('started %s' % stim['name'])

        while self.active:
            try:
                self.log('waiting for message')
                topic, data = self.input_socket.recv_multipart()
                self.log('received message')
                for stim in self.pipeline:
                    if topic in stim['topic'].keys():
                        self.log('running %s'%stim['name'])
                        # call run function with kwargs
                        ret = stim['instance'].run({stim['topic'][topic]: data})
                        self.log('finished {} {}'.format(stim['name'], ret))
            except (KeyboardInterrupt, SystemExit):
                self.active = False
                for init in self.initialization:
                    self.log('stopping %s'%init['name'])
                    init['instance'].stop()
                    self.log('stopped %s'%init['name'])
                for stim in self.pipeline:
                    self.log('stopping %s'%stim['name'])
                    stim['instance'].stop()
                    self.log('stopping %s'%stim['name'])
                sys.exit(0)

class Stimulus(object):
    def __init__(self, **kwargs):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def run(self):
        raise NotImplementedError

class PyCortexViewer(Stimulus):
    def __init__(self, subject, xfm_name, mask_type='thick', vmin=-1., vmax=1., **kwargs):
        super(PyCortexViewer, self).__init__()
        npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()
        
        data = np.random.randn(100, npts)
        # data = np.tile(np.arange(npts), (100,1))
        # data = np.ones((100, npts))
        vol = cortex.Volume(data, subject, xfm_name)

        self.subject = subject
        self.xfm_name = xfm_name
        self.mask_type = mask_type

        self.view = cortex.webshow(vol)
        self.vmin = vmin
        self.vmax = vmax
        self.active = True
        self.i = 0

    def run(self, inp):
        if self.active:
            try:
                data = np.fromstring(inp['data'], dtype=np.float32)
                vol = cortex.Volume(data, self.subject, self.xfm_name, vmin=self.vmin, vmax=self.vmax)
                mos, _ = cortex.mosaic(vol.volume[0])
                self.view.dataviews.data.data[0]._setData(self.i+1, mos*100.)
                self.view.setFrame(self.i)
                self.i += 1

            except IndexError:
                self.active = False
        return 'i={}, data[0]={:.4f}'.format(self.i, data[0])

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
    def __init__(self, **kwargs):
        super(Debug, self).__init__()
    def run(self, inp):
        data = np.fromstring(inp['data'], dtype=np.float32)
        return '{}'.format(len(data))

class AudioRecorder(Stimulus):
    def __init__(self, jack_port, file_name, recording_id, **kwargs):
        super(AudioRecorder, self).__init__()
        rec_path = os.path.join(rec_dir, recording_id, 'logs', file_name+'.wav')
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
    def start(self):
        self.proc = subprocess.Popen(self.cmd)
    def stop(self):
        self.proc.terminate()
        params = [
            {'name': 'input', 'position': 'first', 'value': self.rec_path},
            {'name': 'output', 'position': 'last', 'value': self.rec_path.replace('.wav', '.mp3')}
        ]
        cmd = generate_command('lame', params)
        with open(os.devnull, 'w') as devnull:
            subprocess.call(cmd, stdout=devnull, stderr=devnull)
        os.remove(self.rec_path)
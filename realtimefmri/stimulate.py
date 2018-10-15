#!/usr/bin/env python3
import os
import time
import pickle
import shlex
import subprocess
import numpy as np
import redis
import cortex
from realtimefmri import config


class Stimulus(object):
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def run(self):
        raise NotImplementedError


class Debug(Stimulus):
    def __init__(self, **kwargs):
        super(Debug, self).__init__()

    def run(self, inp):
        data = np.fromstring(inp['data'], dtype='float32')
        return '{}'.format(len(data))


class SendToDashboard(Stimulus):
    """Send data to the dashboard

    Parameters
    ----------
    name : str
    plot_type : str
        Type of plot

    Attributes
    ----------
    redis : redis connection
    key_name : str
        Name of the key in the redis database
    """
    def __init__(self, name, plot_type='marker', host=config.REDIS_HOST, port=6379, **kwargs):
        super(SendToDashboard, self).__init__()
        r = redis.Redis(host=host, port=port)
        key_name = 'dashboard:' + name
        r.set(key_name + ':type', plot_type)

        self.redis = r
        self.key_name = key_name

    def run(self, data):
        self.redis.set(self.key_name, pickle.dumps(data))


class PyCortexViewer(Stimulus):
    bufferlen = 50

    def __init__(self, subject, xfm_name, mask_type='thick', vmin=-1., vmax=1.,
                 **kwargs):
        super(PyCortexViewer, self).__init__()
        npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()

        data = np.zeros((self.bufferlen, npts), 'float32')
        vol = cortex.Volume(data, subject, xfm_name, vmin=vmin, vmax=vmax)
        view = cortex.webshow(vol, autoclose=False)

        self.subject = subject
        self.xfm_name = xfm_name
        self.mask_type = mask_type

        self.view = view
        self.active = True
        self.i = 0

    def update_volume(self, mos):
        i, = self.view.setFrame()
        i = round(i)
        new_frame = (i + 1) % self.bufferlen
        self.view.dataviews.data.data[0]._setData(new_frame, mos)

    def advance_frame(self):
        i, = self.view.setFrame()
        i = round(i)
        self.view.playpause('play')
        time.sleep(1)
        self.view.playpause('pause')
        self.view.setFrame(i + 0.99)

    def run(self, inp):
        if self.active:
            try:
                data = np.fromstring(inp['data'], dtype='float32')
                print(self.subject, self.xfm_name, data.shape)
                vol = cortex.Volume(data, self.subject, self.xfm_name)
                mos, _ = cortex.mosaic(vol.volume[0], show=False)
                self.update_volume(mos)
                self.advance_frame()

                return 'i={}, data[0]={:.4f}'.format(self.i, data[0])

            except IndexError as e:
                self.active = False
                return e

            except Exception as e:
                return e

    def stop(self):
        self.view.playpause('pause')


class RoiBars(Stimulus):
    def __init__(self, **kwargs):
        super(RoiBars, self).__init__()
        raise NotImplementedError


class AudioRecorder(object):
    """Record the microphone and save to file

    Record from the microphone and save as a ``.wav`` file inside of the
    recording folder

    Parameters
    ----------
    jack_port : str
        Name of the jack port
    file_name : str
        Relative path to file name. Will be saved inside the recording folder
    recording_id : str
        Identifier for the recording. Used as the name of the recording folder

    Attributes
    ----------
    rec_path : str
        Path where recording is saves

    Methods
    -------
    start()
        Start the recording
    stop()
        Stop the recording
    """
    def __init__(self, jack_port, file_name, recording_id, **kwargs):
        super(AudioRecorder, self).__init__()
        rec_path = os.path.join(config.RECORDING_DIR, recording_id, file_name + '.wav')
        if not os.path.exists(os.path.dirname(rec_path)):
            os.makedirs(os.path.dirname(rec_path))

        cmd = 'jack_rec -f {} -d {} {}'.format(rec_path, -1, jack_port)

        self.cmd = shlex.split(cmd)
        self.rec_path = rec_path
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(self.cmd)

    def stop(self):
        self.proc.terminate()
        inpath = self.rec_path
        outpath = self.rec_path.replace('.wav', '.mp3')
        cmd = shlex.split('lame {} {}'.format(inpath, outpath))
        with open(os.devnull, 'w') as devnull:
            subprocess.call(cmd, stdout=devnull, stderr=devnull)
        os.remove(self.rec_path)

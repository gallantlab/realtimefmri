#!/usr/bin/env python3
import os
import os.path as op
import importlib
import time
import shlex
import subprocess
import argparse

import yaml
import numpy as np
import zmq
import redis
import cortex

from realtimefmri.utils import get_logger, load_class, parse_message
from realtimefmri.config import RECORDING_DIR, PIPELINE_DIR, STIM_PORT


class Stimulator(object):
    """Highest-level class for running stimulation

    This class reads in the stimulation configuration, connects to the output
    of the preprocessing script, and runs each stimulus as new images arrive.

    Parameters
    ----------
    stim_config : str
        Name of stimulus configuration to use. Should be a file in the
        `pipeline` filestore
    recording_id : str
        A unique identifier for the recording. If none is provided, one will be
        generated from the subject name and date
    in_port : int
        Port number to which data are sent from preprocessing pipeline
    log : bool
        Whether to send log messages to the network log
    verbose : bool
        Whether to log to the console

    Attributes
    ----------
    input_socket : zmq.socket.Socket
        The subscriber socket that receives data sent over from the
        preprocessing script
    active : bool
        Indicates whether the pipeline should be run when data are received
    static_pipeline : dict
        Dictionary that specifies stimulation steps that occurs once at the
        start of the experiment (i.e., it does not receive input with each
        incoming datum)
    pipeline : dict
        Dictionary that specifies stimulation steps that receive each incoming
        datum
    global_parameters : dict
        Parameters that are sent to every stimulation step as keyword arguments

    Methods
    -------
    run()
        Initialize and listen for incoming volumes, processing them as they
        arrive
    """
    def __init__(self, pipeline, global_parameters={}, static_pipeline={}, recording_id=None,
                 in_port=STIM_PORT, log=True, verbose=False):
        """
        """
        super(Stimulator, self).__init__()
        zmq_context = zmq.Context()
        input_socket = zmq_context.socket(zmq.SUB)
        input_socket.connect('tcp://localhost:%d' % in_port)
        input_socket.setsockopt(zmq.SUBSCRIBE, b'')

        if recording_id is None:
            recording_id = 'recording_{}'.format(time.strftime('%Y%m%d_%H%M'))

        self.log = get_logger('stimulate', to_console=verbose, to_network=log)
        self.input_socket = input_socket
        self.active = False

        self.build(pipeline, static_pipeline, global_parameters)

    def build(self, pipeline, static_pipeline, global_parameters):
        """Build the pipeline from the pipeline parameters. Directly sets the class instance 
        attributes for `pipeline`, `static_pipeline`, and `global_parameters`

        Parameters
        ----------
        pipeline : list of dicts
        static_pipeline : list of dicts
        global_parameters : dict
        """
        self.static_pipeline = []
        for step in static_pipeline:
            self.log.debug('initializing %s' % step['name'])
            args = step.get('args', ())
            kwargs = step.get('kwargs', {})
            for k, v in global_parameters.items():
                kwargs[k] = kwargs.get(k, v)

            module = importlib.import_module(step['step'])
            step['instance'] = module(*args, **kwargs)
            self.static_pipeline.append(step)

        self.pipeline = []
        for step in pipeline:
            print(step)
            self.log.debug('initializing %s' % step['name'])
            args = step.get('args', ())
            kwargs = step.get('kwargs', dict())
            for k, v in global_parameters.items():
                kwargs[k] = kwargs.get(k, v)

            cls = load_class(step['step'])
            step['instance'] = cls(*args, **kwargs)
            self.pipeline.append(step)

        self.global_parameters = global_parameters

    @classmethod
    def load_from_saved_pipelines(cls, pipeline_name, **kwargs):
        """Load from the pipelines stored with the pacakge

        Parameters
        ----------
        pipeline_name : str
            The name of a pipeline stored in the pipeline directory

        Returns
        -------
        A Pipeline class for the specified pipeline name
        """
        config_path = op.join(PIPELINE_DIR, pipeline_name + '.yaml')
        return cls.load_from_config(config_path, **kwargs)

    @classmethod
    def load_from_config(cls, config_path, **kwargs):
        with open(config_path, 'rb') as f:
            config = yaml.load(f)

        kwargs.update(config)
        return cls(**kwargs)

    def run(self):
        self.active = True
        self.log.debug('running')

        # run
        for step in self.static_pipeline:
            self.log.debug('starting %s' % step['name'])
            step['instance'].start()
            self.log.debug('started %s' % step['name'])

        for step in self.pipeline:
            self.log.debug('starting %s' % step['name'])
            step['instance'].start()
            self.log.debug('started %s' % step['name'])

        while self.active:
            self.log.debug('waiting for message')
            message = self.input_socket.recv_multipart()
            topic, sync_time, data = parse_message(message)
            self.log.debug('received message')
            for step in self.pipeline:
                if topic in step['topic'].keys():
                    self.log.info('running %s at %s (acq at %s)',
                                  step['name'], time.time(), sync_time)
                    ret = step['instance'].run({step['topic'][topic]: data})
                    self.log.info('finished %s %s',
                                  step['name'], ret)

    def stop(self):
        for step in self.static_pipeline:
            self.log.debug('stopping %s' % step['name'])
            step['instance'].stop()
            self.log.debug('stopped %s' % step['name'])
        for step in self.pipeline:
            self.log.debug('stopping %s' % step['name'])
            step['instance'].stop()
            self.log.debug('stopped %s' % step['name'])


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
    def __init__(self, **kwargs):
        super(SendToDashboard, self).__init__()
        self.redis = redis.Redis()

    def run(self, inp):
        print('send to dashboard')
        inp['name'] = 'volume'
        data = inp['data']
        # if isinstance(data, np.ndarray):
        dtype = 'ndarray'

        self.redis.set('rt_' + inp['name'] + '_data', data)
        self.redis.set('rt_' + inp['name'] + '_dtype', dtype)


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
    '''Record the microphone and save to file

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


    '''
    def __init__(self, jack_port, file_name, recording_id, **kwargs):
        super(AudioRecorder, self).__init__()
        rec_path = os.path.join(RECORDING_DIR, recording_id, file_name + '.wav')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config', action='store',
                        help='Name of configuration file')
    parser.add_argument('recording_id', action='store',
                        help='Recording name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')

    args = parser.parse_args()

    stim = Stimulator.load_from_saved_pipelines(args.config, recording_id=args.recording_id,
                                                verbose=args.verbose)
    try:
        stim.run()  # this will start an infinite run loop
    except KeyboardInterrupt:
        print('shutting down stimulation')
        stim.active = False
        stim.stop()

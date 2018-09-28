import os
import time
import shlex
import subprocess
import argparse

import yaml
import numpy as np
import zmq
import cortex

from realtimefmri.utils import get_logger, parse_message
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
        Whether to send log messages to the network logger
    verbose : bool
        Whether to log to the console

    Attributes
    ----------
    input_socket : zmq.socket.Socket
        The subscriber socket that receives data sent over from the
        preprocessing script
    active : bool
        Indicates whether the pipeline should be run when data are received
    initialization : dict
        Dictionary that specifies stimulation steps that occurs once at the
        start of the experiment (i.e., it does not receive input with each
        incoming datum)
    pipeline : dict
        Dictionary that specifies stimulation steps that receive each incoming
        datum
    global_defaults : dict
        Parameters that are sent to every stimulation step as keyword arguments

    Methods
    -------
    run()
        Initialize and listen for incoming volumes, processing them as they
        arrive
    """
    def __init__(self, stim_config, recording_id=None, in_port=STIM_PORT,
                 log=True, verbose=False):
        """
        """
        super(Stimulator, self).__init__()
        zmq_context = zmq.Context()
        self.input_socket = zmq_context.socket(zmq.SUB)
        self.input_socket.connect('tcp://localhost:%d' % in_port)
        self.input_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.active = False

        with open(os.path.join(PIPELINE_DIR, stim_config + '.yaml'), 'rb') as f:
            config = yaml.load(f)
            self.initialization = config.get('initialization', dict())
            self.pipeline = config['pipeline']
            self.global_defaults = config.get('global_defaults', dict())

        if recording_id is None:
            recording_id = '%s_%s' % (self.global_defaults['subject'],
                                      time.strftime('%Y%m%d_%H%M'))
        self.global_defaults['recording_id'] = recording_id

        self.logger = get_logger('stimulate', to_console=verbose,
                                 to_network=log)

        # initialization
        # initialization pipeline
        for init in self.initialization:
            self.logger.debug('initializing %s' % init['name'])
            params = init.get('kwargs', {})
            for k, v in self.global_defaults.items():
                params.setdefault(k, v)
            init['instance'].__init__(**params)

        # main pipeline
        for step in self.pipeline:
            self.logger.debug('initializing %s' % step['name'])
            params = step.get('kwargs', {})
            for k, v in self.global_defaults.items():
                params.setdefault(k, v)
            step['instance'].__init__(**params)

    def run(self):
        self.active = True
        self.logger.debug('running')

        # run
        for init in self.initialization:
            self.logger.debug('starting %s' % init['name'])
            init['instance'].start()
            self.logger.debug('started %s' % init['name'])

        for stim in self.pipeline:
            self.logger.debug('starting %s' % stim['name'])
            stim['instance'].start()
            self.logger.debug('started %s' % stim['name'])

        while self.active:
            self.logger.debug('waiting for message')
            message = self.input_socket.recv_multipart()
            topic, sync_time, data = parse_message(message)
            self.logger.debug('received message')
            for stim in self.pipeline:
                if topic in stim['topic'].keys():
                    print(topic)
                    self.logger.info('running %s at %s (acq at %s)',
                                     stim['name'], time.time(), sync_time)
                    # call run function with kwargs
                    ret = stim['instance'].run({stim['topic'][topic]: data})
                    self.logger.info('finished %s %s',
                                     stim['name'], ret)

    def stop(self):
        for init in self.initialization:
            self.logger.debug('stopping %s' % init['name'])
            init['instance'].stop()
            self.logger.debug('stopped %s' % init['name'])
        for stim in self.pipeline:
            self.logger.debug('stopping %s' % stim['name'])
            stim['instance'].stop()
            self.logger.debug('stopped %s' % stim['name'])


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


class Debug(Stimulus):
    def __init__(self, **kwargs):
        super(Debug, self).__init__()

    def run(self, inp):
        data = np.fromstring(inp['data'], dtype='float32')
        return '{}'.format(len(data))


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

    stim = Stimulator(args.config, recording_id=args.recording_id,
                      verbose=args.verbose)
    try:
        stim.run()  # this will start an infinite run loop
    except KeyboardInterrupt:
        print('shutting down stimulation')
        stim.active = False
        stim.stop()

#!/usr/bin/env python3
import six
import os
import os.path as op
import struct
import warnings
import time
import json
from uuid import uuid4
import argparse

import yaml
import numpy as np
import zmq

import nibabel as nib

import cortex

from realtimefmri.image_utils import register
from realtimefmri.utils import get_logger, load_class
from realtimefmri.config import (get_subject_directory,
                                 RECORDING_DIR, PIPELINE_DIR,
                                 VOLUME_PORT, PREPROC_PORT)

if six.PY2:
    import cPickle as pickle
    from itertools import izip as zip
    range = xrange
elif six.PY3:
    import pickle


class Preprocessor(object):
    """Highest-level class for running preprocessing

    This class loads the preprocessing pipeline from the configuration
    file, initializes the classes for each step, and runs the main loop
    that receives incoming images from the data collector.

    Parameters
    ----------
    preproc_config : str
        Name of preprocessing configuration to use. Should be a file in the
        `pipeline` filestore
    recording_id : str
        A unique identifier for the recording
    in_port : int
        Port number to which data are sent from data collector
    out_port : int
        Port number to publish preprocessed data to. Stimulation script will
        read from this port
    log : bool
        Whether to send log messages to the network logger
    verbose : bool
        Whether to log to the console


    Attributes
    ----------
    in_port : int
        Port number to which data are sent from data collector.
    input_socket : zmq.socket.Socket
        The subscriber socket that receives data sent over from the data
        collector.
    out_port : int
        Port number to publish preprocessed data to. Stimulation script will
        read from this port.
    output_socket : zmq.socket.Socket
        The publisher socket that sends data over to the stimulator.
    active : bool
        Indicates whether the pipeline should be run when data are received.
    pipeline : dict
        Dictionary that specifies preprocessing steps that receive each
        incoming volume.
    global_parameters : dict
        Parameters that are sent to every stimulation step as keyword
        arguments.
    n_skip : int
        Number of volumes to skip at start of run

    Methods
    -------
    run()
        Initialize and listen for incoming volumes, processing them through the
        pipeline as they arrive
    """
    def __init__(self, preproc_config, recording_id=None, in_port=VOLUME_PORT,
                 out_port=PREPROC_PORT, verbose=False, log=True, **kwargs):
        super(Preprocessor, self).__init__()

        # initialize input and output sockets
        context = zmq.Context()
        self.in_port = in_port
        self.input_socket = context.socket(zmq.SUB)
        self.input_socket.connect('tcp://localhost:%d' % in_port)
        self.input_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.out_port = out_port
        self.output_socket = context.socket(zmq.PUB)
        self.output_socket.bind('tcp://*:%d' % out_port)

        self.active = False

        self.pipeline = Pipeline.load_from_saved_pipelines(preproc_config,
            recording_id=recording_id, output_socket=self.output_socket, log=log, verbose=verbose)

        self.log = get_logger('preprocess', to_console=verbose,
                              to_network=log)

        self.n_skip = self.pipeline.global_parameters.get('n_skip', 0)

    def receive_image(self):
        (_,
         image_id,
         raw_image_nii) = self.input_socket.recv_multipart()
        return image_id, raw_image_nii

    def run(self):
        self.active = True
        self.log.info('running')
        while self.active:
            self.log.debug('waiting for image')
            image_id, raw_image_nii = self.receive_image()
            image_id = struct.unpack('i', image_id)[0]
            self.log.info('received image %d', image_id)
            data_dict = {'image_id': image_id,
                         'raw_image_nii': pickle.loads(raw_image_nii)}
            _ = self.pipeline.process(data_dict)

    def stop(self):
        pass


class Pipeline(object):
    """Construct and run a preprocessing pipeline

    Load a preprocessing configuration file, intialize all of the steps, and
    process each image through the pipeline.

    Parameters
    ----------
    pipeline : list of dict
        The parameters for pipeline steps
    global_parameters : dict
        Settings passed as keyword arguments to each pipeline step
    static_pipeline : list of dict
        Pipeline steps that start with initialization and do not receive input
    recording_id : str
        A unique identifier for the recording. If none is provided, one will be
        generated from the subject name and date
    output_socket : zmq.socket.Socket
        Socket (zmq.PUB) to which preprocessed outputs are sent
    log : bool
        Log to network logger
    verbose : bool
        Log to console

    Attributes
    ----------
    global_parameters : dict
        Dictionary of arguments that are sent as keyword arguments to every
        preprocessing step. Useful for values that are required in multiple
        steps like recording identifier, subject name, and transform name
    static_pipeline : list
        List of dictionaries that configure initialization steps. These are run
        once at the outset of the program.
    pipeline : list
        List of dictionaries that configure steps in the pipeline. These are
        run for each image that arrives at the pipeline.
    log : logging.Logger
        The logger object
    output_socket : zmq.socket.Socket
        Socket (zmq.PUB) to which preprocessed outputs are sent

    Methods
    -------
    process(data_dict)
        Run the data in ```data_dict``` through each of the preprocessing steps
    """
    def __init__(self, pipeline, global_parameters={}, static_pipeline={}, recording_id=None,
                 log=False, output_socket=None, verbose=False):
        if recording_id is None:
            recording_id = 'recording_{}'.format(time.strftime('%Y%m%d_%H%M'))

        log = get_logger('preprocess.pipeline',
                         to_console=verbose,
                         to_network=log)

        self.recording_id = recording_id
        self.log = log
        self.output_socket = output_socket

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

            cls = load_class(step['step'])
            step['instance'] = cls(*args, **kwargs)
            self.static_pipeline.append(step)

        self.pipeline = []
        for step in pipeline:
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

    def process(self, data_dict):
        """Run through the preprocessing steps

        Iterate through all the preprocessing steps. For each step, extract the `input` keys from 
        the `data_dict` ans pass them as ordered unnamed arguments to that step. The return value 
        is saved to the `data_dict` using the  `output` key.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing all the processing results


        Returns
        -------
        A dictionary of all processing results
        """
        image_id = struct.pack('i', data_dict['image_id'])
        for step in self.pipeline:
            args = [data_dict[k] for k in step['input']]

            self.log.debug('running %s' % step['name'])
            outp = step['instance'].run(*args)

            self.log.debug('finished %s' % step['name'])

            if not isinstance(outp, (list, tuple)):
                outp = [outp]

            d = dict(zip(step.get('output', []), outp))
            data_dict.update(d)

            for topic in step.get('send', []):
                message = d[topic]
                self.log.debug('sending %s' % topic)
                if isinstance(message, dict):
                    message = json.dumps(message)
                elif isinstance(message, (np.ndarray)):
                    message = message.astype(np.float32).tostring()
                elif isinstance(message, str):
                    message = message.encode()
                else:
                    print("Type {} not implemented (topic={})."
                          .format(type(message), topic))
                self.output_socket.send_multipart([topic.encode(),
                                                   image_id,
                                                   message])

        return data_dict


class PreprocessingStep(object):
    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError


def load_mask(subject, xfm_name, mask_type):
        mask_path = op.join(cortex.database.default_filestore,
                            subject, 'transforms', xfm_name,
                            'mask_' + mask_type + '.nii.gz')
        return nib.load(mask_path)


def load_reference(subject, xfm_name):
        ref_path = op.join(cortex.database.default_filestore,
                           subject, 'transforms', xfm_name,
                           'reference.nii.gz')
        return nib.load(ref_path)


class Debug(object):
    def __init__(self, **kwargs):
        pass

    def run(self, volume):
        print(volume.shape, 'a volume!')
        return volume.get_data(), volume.shape, 'a volume!'


class SaveNifti(PreprocessingStep):
    """Saves nifti images to files

    Creates a subfolder in the recording directory and saves each incoming
    image as a nifti file.

    Parameters
    ----------
    recording_id : str
        Unique identifier for the run
    path_format : str
        Filename formatting string that is compatible with "%" string
        formatting. Must be able to format an integer containing the TR number.

    Attributes
    ----------
    recording_id : str
        Unique identifier for the run
    path_format : str
        Filename formatting string that is compatible with "%" string
        formatting. Must be able to format an integer containing the TR number.

    Methods
    --------
    run(inp)
        Saves the input image to a file and iterates the counter.
    """

    def __init__(self, recording_id=None, path_format='volume_{:04}.nii',
                 **kwargs):
        if recording_id is None:
            recording_id = str(uuid4())
        recording_dir = op.join(RECORDING_DIR, recording_id, 'nifti')
        try:
            os.makedirs(recording_dir)
        except OSError:
            pass

        print(recording_dir)
        self.recording_dir = recording_dir
        self.path_format = path_format

    def run(self, inp, image_number):
        path = self.path_format.format(image_number)
        nib.save(inp, op.join(self.recording_dir, path))
        print('saving to {}'.format(op.join(self.recording_dir, path)))


class MotionCorrect(PreprocessingStep):
    """Motion corrects images to a reference image

    Uses AFNI ``3dvolreg`` to motion correct the incoming images to a reference
    image stored in the pycortex database.

    Parameters
    ----------
    subject : str
        Subject name in pycortex filestore
    xfm_name : str
        Transform name for the subject in pycortex filestore

    Attributes
    ----------
    reference_affine : numpy.ndarray
        Affine transform for the reference image
    reference_path : str
        Path to the reference image

    Methods
    -------
    run(input_volume)
        Motion corrects the incoming image to the provided reference image and
        returns the motion corrected volume
    """
    def __init__(self, subject, xfm_name, twopass=False, **kwargs):
        ref_path = op.join(cortex.database.default_filestore,
                           subject, 'transforms', xfm_name,
                           'reference.nii.gz')

        nii = nib.load(ref_path)
        self.reference_affine = nii.affine
        self.reference_path = ref_path
        self.twopass = twopass
        print(ref_path)

    def run(self, input_volume):
        same_affine = np.allclose(input_volume.affine[:3, :3],
                                  self.reference_affine[:3, :3])
        if not same_affine:
            print(input_volume.affine)
            print(self.reference_affine)
            raise Exception('Input and reference volumes have different affines.')
        return register(input_volume, self.reference_path, twopass=self.twopass)


class ApplyMask(PreprocessingStep):
    """Apply a voxel mask to the volume.

    Loads a mask from the realtimefmri database. Mask should be in xyz format
    to match data. Mask is applied after transposing mask and data to zyx to
    match the wm detrend training.

    Parameters
    ----------
    subject : str
        Subject name
    xfm_name : str
        Pycortex transform name
    mask_type : str
        Type of mask

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean voxel mask
    mask_affine : numpy.ndarray
        Affine transform for the mask

    Methods
    -------
    run(volume)
        Apply the mask to the input volume
    """
    def __init__(self, subject, xfm_name, mask_type=None, **kwargs):
        mask_path = op.join(cortex.database.default_filestore,
                            subject, 'transforms', xfm_name,
                            'mask_' + mask_type + '.nii.gz')
        self.load_mask(mask_path)
        print(mask_path)

    def load_mask(self, mask_path):
        mask_nifti1 = nib.load(mask_path)
        self.mask_affine = mask_nifti1.affine
        self.mask = mask_nifti1.get_data().astype(bool)

    def run(self, volume):
        same_affine = np.allclose(volume.affine[:3, :3], self.mask_affine[:3, :3])
        assert same_affine, 'Input and mask volumes have different affines.'
        return volume.get_data().T[self.mask.T]


def secondary_mask(mask1, mask2, order='C'):
    """
    Given an array, X and two 3d masks, mask1 and mask2
    X[mask1] = x
    X[mask1 & mask2] = y
    x[new_mask] = y
    """
    assert mask1.shape == mask2.shape
    mask1_flat = mask1.ravel(order=order)
    mask2_flat = mask2.ravel(order=order)

    masks = np.c_[mask1_flat, mask2_flat]
    masks = masks[mask1_flat, :]
    return masks[:, 1].astype(bool)


class ApplyMask2(PreprocessingStep):
    """Apply a second mask to a vector produced by a first mask.

    Given a vector of voxel activity from a primary mask, return voxel activity
    for a secondary mask. Both masks are 3D voxel masks and resulting vector
    will be as if the intersection of primary and secondary masks was applied
    to the original 3D volume.

    Parameters
    ----------
    subject : str
        Subject name
    xfm_name : str
        Pycortex transform name
    mask_type_1 : str
        Mask type for initial mask. Incoming vector results from applying this
        mask to the 3D volume
    mask_type_2 : str
        Mask type for secondary mask.

    Attributes
    ----------
    mask : numpy.ndarray
       A boolean vector that selects elements from the vector output of primary
       mask applied to a volume that are also in secondary mask.

    Methods
    -------
    run(x)
        Returns a vector of voxel activity of the intersection between primary
        and secondary masks
    """
    def __init__(self, subject, xfm_name, mask_type_1, mask_type_2, **kwargs):
        mask1 = cortex.db.get_mask(subject, xfm_name, mask_type_1).T  # in xyz
        mask2 = cortex.db.get_mask(subject, xfm_name, mask_type_2).T  # in xyz
        self.mask = secondary_mask(mask1, mask2, order='F')

    def run(self, x):
        if x.ndim > 1:
            x = x.reshape(-1, 1)
        return x[self.mask]


class ActivityRatio(PreprocessingStep):
    def __init__(self, **kwargs):
        super(ActivityRatio, self).__init__()

    def run(self, x1, x2):
        if isinstance(x1, np.ndarray):
            x1 = np.nanmean(x1)
        if isinstance(x2, np.ndarray):
            x2 = np.nanmean(x2)

        return x1 / (x1 + x2)


class RoiActivity(PreprocessingStep):
    """Extract activity from an ROI.

    Placeholder

    Parameters
    ----------
    subject : str
        subject ID
    xfm_name : str
        pycortex transform ID
    pre_mask_name : str
        ROI masks returned by pycortex are in volume space, but activity is
        provided as a vector of gray matter activity. ``pre_mask_name`` is the
        name of the mask that was applied to the raw image volume to produce
        the gray matter activity vector.
    roi_names : list of str
        names of the ROIs to extract

    Attributes
    ----------
    masks : dict
        A dictionary containing the voxel masks for each named ROI

    Methods
    -------
    run():
        Returns a list of floats of mean activity in the requested ROIs
    """
    def __init__(self, subject, xfm_name, pre_mask_name, roi_names, **kwargs):

        subj_dir = get_subject_directory(subject)
        pre_mask_path = op.join(subj_dir, pre_mask_name+'.nii')

        # mask in zyx
        pre_mask = nib.load(pre_mask_path).get_data().T.astype(bool)

        # returns masks in zyx
        roi_masks, roi_dict = cortex.get_roi_masks(subject, xfm_name, roi_names)

        self.masks = dict()
        for name, mask_value in roi_dict.items():
            roi_mask = roi_masks == mask_value
            self.masks[name] = secondary_mask(pre_mask, roi_mask)

    def run(self, activity):
        if activity.ndim > 1:
            activity = activity.reshape(-1, 1)
        roi_activities = dict()
        for name, mask in self.masks.items():
            roi_activities[name] = float(activity[mask].mean())
        return roi_activities


class WMDetrend(PreprocessingStep):
    """Detrend a volume using white matter detrending

    Uses a pre-trained white matter detrender to remove the trend from a
    volume.

    Parameters
    ----------
    subject : str
        Subject identifier
    model_name : str
        Name of the pre-trained white matter detrending model

    Attributes
    ----------
    subject : str
        Subject identifier
    model_name : str
        Name of white matter detrending model in subject's directory

    Methods
    -------
    run(wm_activity, gm_activity)
        Returns detrended grey matter activity given raw gray and white matter
        activity
    """
    def __init__(self, subject, model_name=None, **kwargs):
        """
        """
        super(WMDetrend, self).__init__()
        subj_dir = get_subject_directory(subject)

        model_path = op.join(subj_dir, 'model-%s.pkl' % model_name)
        pca_path = op.join(subj_dir, 'pca-%s.pkl' % model_name)

        with open(model_path, 'r') as f:
            model = pickle.load(f)

        with open(pca_path, 'r') as f:
            pca = pickle.load(f)

        self.model = model
        self.pca = pca

    def run(self, wm_activity, gm_activity):
        wm_activity_pcs = self.pca.transform(wm_activity.reshape(1, -1)).reshape(1, -1)
        gm_trend = self.model.predict(wm_activity_pcs)
        return gm_activity - gm_trend


class RunningMeanStd(PreprocessingStep):
    """Compute a running mean and standard deviation for a set of voxels

    Compute a running mean and standard deviation, looking back a set number of
    samples.

    Parameters
    ----------
    n : int
        The number of past samples over which to compute mean and standard
        deviation

    Attributes
    ----------
    n : int
        The number of past samples over which to compute mean and standard
        deviation
    mean : numpy.ndarray
        The mean for the samples
    std : numpy.ndarray
        The standard deviation for the samples
    samples : numpy.ndarray
        The stored samples

    Methods
    -------
    run(inp)
        Adds the input vector to the stored samples (discard the oldest sample)
        and compute and return the mean and standard deviation.
    """
    def __init__(self, n=20, skip=5, **kwargs):
        self.n = n
        self.mean = None
        self.samples = None
        self.skip = skip

    def run(self, inp, image_number=None):
        if image_number < self.skip:
            return np.zeros(inp.size), np.ones(inp.size)

        if self.mean is None:
            self.samples = np.empty((self.n, inp.size))*np.nan
        else:
            self.samples[:-1, :] = self.samples[1:, :]
        self.samples[-1, :] = inp
        self.mean = np.nanmean(self.samples, 0)
        self.std = np.nanstd(self.samples, 0)
        return self.mean, self.std


class VoxelZScore(PreprocessingStep):
    """Compute a z-score of a vector

    Z-score a vector given precomputed mean and standard deviation

    Attributes
    ----------
    mean : numpy.ndarray
        A vector of voxel means
    std : numpy.ndarray
        A vector of voxel standard deviations

    Methods
    -------
    zscore(data)
        Apply the mean and standard deviation to compute and return a z-scored
        version of the data
    run(inp, mean=None, std=None)
        Return the z-scored version of the data
    """
    def __init__(self, **kwargs):
        self.mean = None
        self.std = None

    def zscore(self, data):
        return (data - self.mean) / self.std

    def run(self, inp, mean=None, std=None):
        # update mean and std if provided
        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std

        # zscore
        if self.mean is None:
            z = inp
        else:
            if (self.std == 0).all():
                z = np.zeros_like(inp)
            else:
                z = self.zscore(inp)
        return z


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config',
                        action='store',
                        help='Name of configuration file')
    parser.add_argument('recording_id', action='store',
                        help='Recording name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')

    args = parser.parse_args()

    preproc = Preprocessor(args.config, recording_id=args.recording_id,
                           verbose=args.verbose)
    try:
        preproc.run()
    except KeyboardInterrupt:
        print('shutting down preprocessing')
        preproc.active = False
        preproc.stop()

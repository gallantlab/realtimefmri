#!/usr/bin/env python
import six
import os
import os.path as op
if six.PY2:
    import cPickle as pickle
    from itertools import izip as zip
    range = xrange
elif six.PY3:
    import pickle
from glob import glob
import time
import json
import warnings
from uuid import uuid4
import argparse

import yaml
import numpy as np
import zmq

from nibabel import save as nbsave, load as nbload
from nibabel.nifti1 import Nifti1Image

import cortex

from realtimefmri.image_utils import transform, mosaic_to_volume
from realtimefmri.utils import get_logger
from realtimefmri.config import (get_subject_directory,
                                 RECORDING_DIR, PIPELINE_DIR)

VOLUME_PORT = 5557
PREPROC_PORT = 5558


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
    global_defaults : dict
        Parameters that are sent to every stimulation step as keyword
        arguments.
    nskip : int
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

        self.pipeline = Pipeline(preproc_config, recording_id,
                                 output_socket=self.output_socket,
                                 log=log, verbose=verbose)

        self.logger = get_logger('preprocessing', to_console=verbose,
                                 to_network=log)

        self.nskip = self.pipeline.global_defaults.get('nskip', 0)

    def receive_image(self):
        (_,
         raw_image_id,
         raw_image_binary) = self.input_socket.recv_multipart()
        return raw_image_id, raw_image_binary

    def run(self):
        self.active = True
        self.logger.info('running')
        while self.active:
            self.logger.debug('waiting for image')
            raw_image_id, raw_image_binary = self.receive_image()
            self.logger.info('received image %s', raw_image_id)
            data_dict = {'raw_image_id': raw_image_id,
                         'raw_image_binary': raw_image_binary}
            _ = self.pipeline.process(data_dict)

    def stop(self):
        pass


class Pipeline(object):
    """Construct and run a preprocessing pipeline

    Load a preprocessing configuration file, intialize all of the steps, and
    process each image through the pipeline.

    Parameters
    ----------
    config : str
        Path to the preprocessing configuration file
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
    global_defaults : dict
        Dictionary of arguments that are sent as keyword arguments to every
        preprocessing step. Useful for values that are required in multiple
        steps like recording identifier, subject name, and transform name
    initialization : list
        List of dictionaries that configure initialization steps. These are run
        once at the outset of the program.
    steps : list
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
    def __init__(self, config, recording_id=None, log=False,
                 output_socket=None, verbose=False):

        log = get_logger('preprocess.pipeline',
                         to_console=verbose,
                         to_network=log)

        self._from_path(config)
        if recording_id is None:
            recording_id = '%s_%s' % (self.global_defaults['subject'],
                                      time.strftime('%Y%m%d_%H%M'))
        self.global_defaults['recording_id'] = recording_id

        self.log = log
        self.output_socket = output_socket

        for init in self.initialization:
            args = init.get('args', ())
            kwargs = init.get('kwargs', {})
            self.log.debug('initializing %s' % init['name'])
            for k, v in self.global_defaults.items():
                params.setdefault(k, v)
            init['instance'].__init__(*args, **kwargs)

        for step in self.steps:
            self.log.debug('initializing %s' % step['name'])
            args = step.get('args', ())
            kwargs = step.get('kwargs', dict())
            for k, v in self.global_defaults.items():
                kwargs.setdefault(k, v)
            step['instance'].__init__(*args, **kwargs)

    def _from_path(self, preproc_config):
        # load the pipeline from pipelines.conf
        with open(op.join(PIPELINE_DIR, preproc_config+'.yaml'), 'rb') as f:
            self._from_file(f)

    def _from_file(self, f):
        config = yaml.load(f)
        self.initialization = config.get('initialization', [])
        self.steps = config['pipeline']
        self.global_defaults = config.get('global_defaults', dict())

    def process(self, data_dict):
        raw_image_id = data_dict['raw_image_id']
        
        for step in self.steps:
            args = [data_dict[i] for i in step['input']]
            
            self.log.debug('running %s' % step['name'])
            outp = step['instance'].run(*args)
            
            self.log.debug('finished %s' % step['name'])
            
            if not isinstance(outp, (list, tuple)):
                outp = [outp]
            
            d = dict(zip(step.get('output', []), outp))
            data_dict.update(d)
            
            for topic in step.get('send', []):
                self.log.debug('sending %s' % topic)
                if isinstance(d[topic], dict):
                    msg = json.dumps(d[topic])
                elif isinstance(d[topic], (np.ndarray)):
                    msg = d[topic].astype(np.float32).tostring()
                
                self.output_socket.send_multipart([topic.encode(),
                                                   raw_image_id,
                                                   msg])

        return data_dict


class PreprocessingStep(object):
    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError


def load_mask(subject, xfm_name, mask_type):
        mask_path = op.join(cortex.database.default_filestore,
                            subject, 'transforms', xfm_name,
                            'mask_'+mask_type+'.nii.gz')
        return nbload(mask_path)


def load_reference(subject, xfm_name):
        ref_path = op.join(cortex.database.default_filestore,
                           subject, 'transforms', xfm_name,
                           'reference.nii.gz')
        return nbload(ref_path)


class RawToNifti(PreprocessingStep):
    """Converts a mosaic image to a nifti image.

    Takes a 600 x 600 mosaic image of ``uint16`` and turns it into a volume.
    Applies the affine provided from the given transform name.

    Parameters
    ----------
    subject : str
        Subject identifier
    xfm_name : str
        Pycortex transform name

    Attributes
    ----------
    affine : numpy.ndarray
        Affine transform

    Methods
    -------
    run(inp)
        Returns a nifti image of the raw data using the provided affine
        transform

    """
    def __init__(self, subject, xfm_name, **kwargs):
        self.affine = load_reference(subject, xfm_name).affine

    def run(self, inp):
        '''
        pixel_image is a binary string loaded directly from the .PixelData file
        saved on the scanner console

        returns a nifti1 image of the same data in xyz
        '''
        # siements mosaic format is strange
        mosaic = np.fromstring(inp, dtype='uint16').reshape(600, 600,
                                                            order='C')
        # axes 0 and 1 must be swapped because mosaic is PLS and we need LPS
        # voxel data (affine values are -/-/+ for dimensions 1-3, yielding RAS)
        # we want the voxel data orientation to match that of the functional
        # reference, gm, and wm masks
        volume = mosaic_to_volume(mosaic).swapaxes(0, 1)[..., :30]
        return Nifti1Image(volume, self.affine)


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

    def __init__(self, recording_id=None, path_format='volume_%4.4d.nii',
                 **kwargs):
        if recording_id is None:
            recording_id = str(uuid4())
        self.recording_dir = op.join(RECORDING_DIR, recording_id, 'nifti')
        self.path_format = path_format
        self._i = 0
        try:
            os.makedirs(self.recording_dir)
        except OSError:
            self._i = self._infer_i()
            warnings.warn('''Save directory already exists. Beginning file
                             numbering with %d''' % self._i)

    def _infer_i(self):
        from re import compile as re_compile
        pattern = re_compile("\%[0-9]*\.?[0-9]*[uifd]")
        match = pattern.split(self.path_format)
        glob_pattern = '*'.join(match)
        fpaths = glob(op.join(self.recording_dir, glob_pattern))

        i_pattern = re_compile('(?<={})[0-9]*(?={})'.format(*match))
        try:
            max_i = max([int(i_pattern.findall(i)[0]) for i in fpaths])
            i = max_i + 1
        except ValueError:
            i = 0
        return i

    def run(self, inp):
        fpath = self.path_format % self._i
        nbsave(inp, op.join(self.recording_dir, fpath))
        self._i += 1


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
    def __init__(self, subject, xfm_name, **kwargs):
        ref_path = op.join(cortex.database.default_filestore,
                           subject, 'transforms', xfm_name,
                           'reference.nii.gz')

        nii = nbload(ref_path)
        self.reference_affine = nii.affine
        self.reference_path = ref_path

    def run(self, input_volume):
        assert np.allclose(input_volume.affine, self.reference_affine)
        return transform(input_volume, self.reference_path)


class ApplyMask(PreprocessingStep):
    '''Apply a voxel mask to the volume.

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
    '''
    def __init__(self, subject, xfm_name, mask_type=None, **kwargs):
        mask_path = op.join(cortex.database.default_filestore,
                            subject, 'transforms', xfm_name,
                            'mask_'+mask_type+'.nii.gz')
        self.load_mask(mask_path)

    def load_mask(self, mask_path):
        mask_nifti1 = nbload(mask_path)
        self.mask_affine = mask_nifti1.affine
        self.mask = mask_nifti1.get_data().astype(bool)

    def run(self, volume):
        assert np.allclose(volume.affine, self.mask_affine)
        return volume.get_data().T[self.mask.T]


def secondary_mask(mask1, mask2, order='C'):
    '''
    Given an array, X and two 3d masks, mask1 and mask2
    X[mask1] = x
    X[mask1 & mask2] = y
    x[new_mask] = y
    '''
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

        return x1/(x1+x2)


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
        pre_mask = nbload(pre_mask_path).get_data().T.astype(bool)

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
    volume. This should set up the class instance to be ready to take an image
    input and output the detrended gray matter activation. To do this, it needs
    ``wm_mask_funcref``, the white matter masks in functional reference space,
    ``gm_mask_funcref``, the grey matter masks in functional reference space,
    ``funcref_nifti1``, the functional reference image, ``input_affine``,
    affine transform for the input images. Since we'll be dealing withraw pixel
    data, we need to have a predetermined image orientation.

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
    subj_dir : str
        Path to the subject's filestore
    funcref_nifti1 : nibabel.NiftiImage1
        Functional space reference image
    model : sklearn.linear_regression.LinearRegression
        Linear regression that predicts grey matter trend from white matter
        activity
    pca : sklearn.decomposition.PCA
        Principal component analysis that decomposes white matter activity into
        principal components

    Methods
    -------
    run(wm_activity, gm_activity)
        Returns detrended grey matter activity given raw gray and white matter
        activity
    """
    def __init__(self, subject, model_name=None, **kwargs):
        '''
        '''
        super(WMDetrend, self).__init__()
        self.subj_dir = get_subject_directory(subject)
        self.subject = subject

        self.funcref_nifti1 = nbload(op.join(self.subj_dir, 'funcref.nii'))

        try:
            model_path = op.join(self.subj_dir, 'model-%s.pkl' % model_name)
            pca_path = op.join(self.subj_dir, 'pca-%s.pkl' % model_name)

            with open(model_path, 'r') as f:
                model = pickle.load(f)
            self.model = model

            with open(pca_path, 'r') as f:
                pca = pickle.load(f)
            self.pca = pca
        except IOError:
            warnings.warn(('''Could not load...\n\tModel from %s\nand\n\tPCA
                              from %s. Load them manually before running.'''
                              % (model_path, pca_path)))

    def run(self, wm_activity, gm_activity):
        wm_activity_pcs = self.pca.transform(wm_activity.reshape(1, -1)).reshape(1, -1)
        gm_trend = self.model.predict(wm_activity_pcs)
        return gm_activity - gm_trend


def compute_raw2var(raw1, raw2, *args):
    '''Use the raw moments to compute the 2nd central moment
    VAR(X) = E[X^2] - E[X]^2
    '''
    return raw2 - raw1**2


def compute_raw2skew(raw1, raw2, raw3, *args):
    '''Use the raw moments to compute the 3rd standardized moment
    Skew(X) = (E[X^3] - 3*E[X]*E[X^2] + 2*E[X]^3)/VAR(X)^(3/2)
    '''
    # get central moments
    cm2 = raw2 - raw1**2
    cm3 = raw3 - 3*raw1*raw2 + 2*raw1**3
    # get standardized 3rd moment
    sm3 = cm3/cm2**1.5
    return sm3

def compute_raw2kurt(raw1, raw2, raw3, raw4, *args):
    '''Use the raw moments to compute the 4th standardized moment
    Kurtosis(X) = (E[X^4] - 4*E[X]*E[X^3] + 6*E[X]^2*E[X^2] - 3*E[X]^4)/VAR(X)^2 - 3.0
    '''
    # get central moments
    cm2 = raw2 - raw1**2
    cm4 = raw4 - 4*raw1*raw3 + 6*(raw1**2)*raw2 - 3*raw1**4
    # get standardized 4th moment
    sm4 = cm4/cm2**2 - 3
    return sm4

def convert_parallel2moments(node_raw_moments, nsamples):
    '''Combine the online parallel computations of
    `online_moments` objects to compute moments.

    Parameters
    -----------
    node_raw_moments : list
        Each element in the list is a node output.
        Each node output is a `len(4)` object where the
        Nth element is Nth raw moment sum: `np.sum(X**n)`.
        Node moments are aggregated across nodes to compute:
        mean, variance, skewness, and kurtosis.
    nsamples : scalar
        The total number of samples across nodes


    Returns
    -------
    mean : array-like
        The mean of the full distribution
    variance: array-like
        The variance of the full distribution
    skewness: array-like
        The skewness of the full distribution
    kurtosis: array-like
        The kurtosis of the full distribution
    '''
    mean_moments = []
    for raw_moment in zip(*node_raw_moments):
        moment = np.sum(raw_moment, 0)/nsamples
        mean_moments.append(moment)

    emean = mean_moments[0]
    evar = compute_raw2var(*mean_moments)
    eskew = compute_raw2skew(*mean_moments)
    ekurt = compute_raw2kurt(*mean_moments)
    return emean, evar, eskew, ekurt


class OnlineMoments(PreprocessingStep):
    """Compute 1-Nth raw moments online

    For the Ith moment: E[X^i] = (1/n)*\Sum(X^i). This function only stores
    \Sum(X^i) and keeps track of the number of observations.

    Parameters
    ----------
    order : int
        The number of moments to compute

    Attributes
    ----------
    order : int
        The number of moments to compute
    all_raw_moments : numpy.ndarray
        All of the raw moments

    Methods
    -------
    update(x)
        Update the moments given the new observations
    get_statistics()
        Compute the statistics for the data
    get_raw_moments()
        Return the raw moments
    get_norm_raw_moments
        Return normalized raw moments
    run(inp)
        Return the mean and standard deviation
    """
    def __init__(self, order=4, **kwargs):
        self.n = 0.0
        self.order = order
        self.all_raw_moments = [0.0]*self.order
        for odx in range(self.order):
            self.__setattr__('rawmnt%i'%(odx+1), self.all_raw_moments[odx])

    def __repr__(self):
        return '%s.online_moments <%i samples>' % (__name__, int(self.n))

    def update(self, x):
        '''Update the raw moments

        Parameters
        ----------
        x : np.ndarray, or scalar-like
            The new observation. This can be any dimension.
        '''
        self.n += 1

        for odx in range(self.order):
            name = 'rawmnt%i'%(odx+1)
            self.all_raw_moments[odx] = self.all_raw_moments[odx] + x**(odx+1)
            self.__setattr__(name, self.all_raw_moments[odx])

    def get_statistics(self):
        '''Return the 1,2,3,4-moment estimates'''
        # mean,var,skew,kurt
        return convert_parallel2moments([self.get_raw_moments()[:4]],
                                        self.n)

    def get_raw_moments(self):
        return self.all_raw_moments

    def get_norm_raw_moments(self):
        return map(lambda x: x/float(self.n), self.all_raw_moments)

    def run(self, inp):
        self.update(inp)
        return self.get_statistics()[:2]


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
    def __init__(self, n=20, **kwargs):
        self.n = n
        self.mean = None
        self.samples = None
    def run(self, inp):
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
        return (data-self.mean)/self.std

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

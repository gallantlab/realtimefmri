#!/usr/bin/env python3
import os
import os.path as op
import pickle
import time
import warnings
from uuid import uuid4

import dash_core_components as dcc
import dash_html_components as html
import nibabel as nib
import numpy as np
import redis
import yaml

import cortex

from datetime import datetime

from realtimefmri import buffered_array, config, image_utils, pipeline_utils
from realtimefmri.utils import get_logger

logger = get_logger('realtimefmri.preprocess', to_console=True, to_network=False, to_file=True)
r = redis.StrictRedis(config.REDIS_HOST)


def preprocess(recording_id, pipeline_name, **global_parameters):
    """Highest-level class for running preprocessing

    This class loads the preprocessing pipeline from the configuration
    file, initializes the classes for each step, and runs the main loop
    that receives incoming images from the data collector.

    Parameters
    ----------
    pipeline_name : str
        Name of preprocessing configuration to use. Should be a file in the
        `pipeline` filestore
    recording_id : str
        A unique identifier for the recording
    log : bool
        Whether to send log messages to the network logger
    verbose : bool
        Whether to log to the console
    """
    config_path = op.join(config.PIPELINE_DIR, pipeline_name + '.yaml')
    with open(config_path, 'rb') as f:
        pipeline_config = yaml.load(f)

    pipeline_config['global_parameters'].update(global_parameters)

    pipeline = Pipeline(**pipeline_config)

    # XXX: global n_skip is unused.
    # n_skip = pipeline.global_parameters.get('n_skip', 0)

    volume_subscription = r.pubsub()
    volume_subscription.subscribe('timestamped_volume')
    volume_subscription.subscribe('pipeline_reset')
    for message in volume_subscription.listen():
        if message['channel'] == b'timestamped_volume' and message['type'] == 'message':
            timestamped_volume = pickle.loads(message['data'])
            logger.info('Received image %d', timestamped_volume['image_number'])
            data_dict = {'image_number': timestamped_volume['image_number'],
                         'raw_image_time': timestamped_volume['time'],
                         'raw_image_nii': timestamped_volume['volume']}
            cue = r.get("cur_cue")
            if cue is not None:
                data_dict['experiment_info'] = dict(cur_cue=cue.decode('utf-8'))
            else:
                data_dict['experiment_info'] = None

            t1 = time.time()
            data_dict = pipeline.process(data_dict)
            t2 = time.time()
            logger.debug('Pipeline ran in %.4f seconds', t2 - t1)

        elif message['channel'] == b'pipeline_reset' and message['type'] == 'message':
            pipeline.reset()
            logger.info('Pipeline reset.')


class Pipeline():
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

    Methods
    -------
    process(data_dict)
        Run the data in ```data_dict``` through each of the preprocessing steps
    """
    def __init__(self, pipeline, static_pipeline=None, global_parameters=None, recording_id=None):
        if recording_id is None:
            recording_id = 'recording_{}'.format(time.strftime('%Y%m%d_%H%M'))

        self.recording_id = recording_id
        self.global_parameters = global_parameters
        self.static_pipeline = None  # set in self.build
        self.pipeline = None  # set in self.build

        self.build(pipeline, static_pipeline)
        self.register()

    def _build_static_pipeline(self, static_pipeline_steps):
        """Build the static pipeline

        Parameters
        ----------
        static_pipeline : list of dict or None
            List of dictionaries that configure initialization steps. These are run
            once at the outset of the program.
        global_parameters : dict or None
        """
        if static_pipeline_steps:
            static_pipeline = []
            for step in static_pipeline_steps:
                logger.debug('Initializing %s', step['name'])
                args = step.get('args', ())
                kwargs = step.get('kwargs', {})

                if self.global_parameters:
                    for key in self.global_parameters.keys():
                        if key not in kwargs:
                            kwargs[key] = self.global_parameters[key]

                cls = pipeline_utils.load_class(step['class_name'])
                step['instance'] = cls(*args, **kwargs)
                static_pipeline.append(step)

        else:
            static_pipeline = []

        self.static_pipeline = static_pipeline

    def _build_pipeline(self, pipeline_steps):
        """Build the pipeline from pipeline parameters

        Parameters
        ----------
        pipeline : list of dicts
        """
        pipeline = []
        for step in pipeline_steps:
            logger.debug('Initializing %s', step['name'])
            args = step.get('args', ())
            kwargs = step.get('kwargs', dict())

            if self.global_parameters:
                for k, v in self.global_parameters.items():
                    kwargs[k] = kwargs.get(k, v)

            cls = pipeline_utils.load_class(step['class_name'])
            step['instance'] = cls(*args, **kwargs)
            pipeline.append(step)

        self.pipeline = pipeline

    def build(self, pipeline, static_pipeline):
        """Build the pipeline from the pipeline parameters. Directly sets the class instance
        attributes for `pipeline` and `static_pipeline`

        Parameters
        ----------
        pipeline : list of dicts
        static_pipeline : list of dicts
        """
        self._build_static_pipeline(static_pipeline)
        self._build_pipeline(pipeline)

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
        config_path = op.join(config.PIPELINE_DIR, pipeline_name + '.yaml')
        return cls.load_from_config(config_path, **kwargs)

    @classmethod
    def load_from_config(cls, config_path, **kwargs):
        with open(config_path, 'rb') as f:
            conf = yaml.load(f)

        kwargs.update(conf)
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
        for step in self.pipeline:
            inputs = [data_dict[k] for k in step['input']]

            logger.info('Running %s', step['name'])
            t1 = time.time()
            outp = step['instance'].run(*inputs)
            t2 = time.time()
            logger.debug('Step %s %s ran in %.4f seconds', step['name'], str(outp), t2 - t1)

            if not isinstance(outp, (list, tuple)):
                outp = [outp]

            d = dict(zip(step.get('output', []), outp))
            logger.debug('Updating data dict with %s', str(d))
            data_dict.update(d)

        return data_dict

    @staticmethod
    def create_interface(key):
        contents = []
        for class_name_key in r.scan_iter(key + b':*:class_name'):
            class_name = pickle.loads(r.get(class_name_key))
            step_index = int(class_name_key.split(b':')[2].decode('utf-8'))
            step_class = pipeline_utils.load_class(class_name)
            step_key = class_name_key.rsplit(b':', maxsplit=1)[0]
            interface = step_class.interface(step_key)
            contents.append([step_index, interface])

        contents = sorted(contents, key=lambda x: x[0])
        contents = [content for i, content in contents]

        return contents

    def register(self):
        """Register the pipeline to the redis database
        """
        pipeline_key = f'pipeline:{id(self)}'
        logger.debug('Registering pipeline %s', pipeline_key)
        for step_index, step in enumerate(self.pipeline):
            step_key = f'{pipeline_key}:{step_index}'
            step['instance'].register(step_key)

        self._key = pipeline_key

    def reset(self):
        """Reset internal states of each preprocessing step
        """
        for step in self.pipeline:
            step['instance'].reset()


class PreprocessingStep():
    def __init__(self, *args, **kwargs):
        self._parameters = kwargs

    def register(self, key):
        """Register the preprocessing step to the redis database

        Parameters
        ----------
        key : str
            A unique key for this step. Convention is pipeline:<pipeline_id>:<step_index>,
            e.g., pipeline:105874924:0, pipeline:105874924:1, pipeline:105874924:2, etc.
        """
        step_name = pipeline_utils.get_step_name(self.__class__)
        r.set(key + ':class_name', pickle.dumps(step_name))
        logger.debug('Registering step %s', step_name)

        for k, v in self._parameters.items():
            r.set(key + f':{k}', pickle.dumps(v))

        self._key = key

    @staticmethod
    def interface(step_key):
        """Define an interface element for the control panel
        """
        step_id = step_key.decode('utf-8').replace(':', '-')

        step = {}
        for key in r.scan_iter(step_key + b':*'):
            param_name = key.rsplit(b':', maxsplit=1)[1]
            val = r.get(key)
            step[param_name] = pickle.loads(val)

        name = step.pop(b'class_name')
        contents = [html.H3(step_key.decode('utf-8')), html.H3(name)]
        for k, v in step.items():
            k = k.decode('utf-8')
            contents.extend([html.Strong(k),
                             dcc.Input(value=v, id=f'{step_id}-{k}')])

        interface = html.Div(contents, id=f'{step_id}')
        logger.debug(interface)
        return interface

    def reset(self):
        pass

    def update_state(self):
        for k in self._parameters.keys():
            v = r.get(self._key + f':{k}')
            v = pickle.loads(v)
            logger.debug(f'Setting {k} to {v}')
            setattr(self, k, v)

    def run(self, *args):
        raise NotImplementedError


class Debug(PreprocessingStep):
    def run(self, nii):
        return str(nii), nii.shape


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

    def __init__(self, *args, recording_id=None, path_format='volume_{:04}.nii', **kwargs):
        parameters = {'recording_id': recording_id, 'path_format': path_format}
        parameters.update(kwargs)
        super(SaveNifti, self).__init__(**parameters)

        if recording_id is None:
            recording_id = str(uuid4())
        recording_dir = op.join(config.RECORDING_DIR, recording_id, 'nifti')
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
    surface : str
        surface name in pycortex filestore
    transform : str
        Transform name for the surface in pycortex filestore

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
    def __init__(self, surface, transform, *args, twopass=False, output_transform=False, **kwargs):
        parameters = {'surface': surface, 'transform': transform, 'twopass': twopass,
                      'output_transform': output_transform}
        parameters.update(kwargs)
        super(MotionCorrect, self).__init__(**parameters)

        reference = cortex.db.get_xfm(surface, transform).reference

        self.reference_affine = reference.affine
        self.reference_path = reference.get_filename()
        self.twopass = twopass
        self.output_transform = output_transform

    def run(self, input_volume):
        same_affine = np.allclose(input_volume.affine[:3, :3],
                                  self.reference_affine[:3, :3])
        if not same_affine:
            logger.info(input_volume.affine)
            logger.info(self.reference_affine)
            warnings.warn('Input and reference volumes have different affines.')

        return image_utils.register(input_volume, self.reference_path,
                                    twopass=self.twopass, output_transform=self.output_transform)


class Function(PreprocessingStep):
    def __init__(self, function_name, *args, **kwargs):
        parameters = {'function_name': function_name}
        parameters.update(kwargs)
        super(Function, self).__init__(**parameters)
        self.function = pipeline_utils.load_class(function_name)

    def run(self, *args):
        return self.function(*args)


class NiftiToVolume(PreprocessingStep):
    """Extract data volume from Nifti image. Translates image dimensions to be consistent with
    pycortex convention, e.g., volume shape is (30, 100, 100)
    """
    def run(self, nii):
        return nii.get_data().T


class VolumeToMosaic(PreprocessingStep):
    def __init__(self, *args, dim=0, **kwargs):
        parameters = {'dim': dim}
        parameters.update(kwargs)
        super(VolumeToMosaic, self).__init__(**parameters)
        self.dim = dim

    def run(self, volume):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cortex.mosaic(volume, dim=self.dim, show=False)[0]


class ApplyMask(PreprocessingStep):
    """Apply a voxel mask from the pycortex database to a volume

    Parameters
    ----------
    surface : str
        Subject name
    transform : str
        Pycortex transform name
    mask_type : str
        Type of mask

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean voxel mask
    """
    def __init__(self, surface, transform, *args, mask_type=None, **kwargs):
        parameters = {'surface': surface, 'transform': transform, 'mask_type': mask_type}
        parameters.update(kwargs)
        super(ApplyMask, self).__init__(**parameters)
        mask = cortex.db.get_mask(surface, transform, mask_type)
        self.mask = mask

    def run(self, volume):
        """Apply the mask to a volume

        Parameters
        -----------
        volume : array
        """
        return volume[self.mask]


class ArrayMean(PreprocessingStep):
    """Compute the mean of an array

    Parameters
    ----------
    dimensions : tuple of int
        Dimensions along which to take the mean. None takes the mean of all values in the array
    """
    def __init__(self, dimensions, *args, **kwargs):
        parameters = {'dimensions': dimensions}
        parameters.update(kwargs)
        super(ArrayMean, self).__init__(**parameters)
        self.dimensions = tuple(dimensions)

    def run(self, array):
        """Take the mean of the array along the specified dimensions

        Parameters
        -----------
        array : array
        """
        if self.dimensions is None:
            return np.mean(array)
        else:
            return np.mean(array, axis=self.dimensions)


class ApplySecondaryMask(PreprocessingStep):
    """Apply a second mask to a vector produced by a first mask.

    Given a vector of voxel activity from a primary mask, return voxel activity
    for a secondary mask. Both masks are 3D voxel masks and resulting vector
    will be as if the intersection of primary and secondary masks was applied
    to the original 3D volume.

    Parameters
    ----------
    surface : str
        Subject name
    transform : str
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
    def __init__(self, surface, transform, mask_type_1, mask_type_2, **kwargs):
        parameters = {'surface': surface, 'transform': transform,
                      'mask_type_1': mask_type_1, 'mask_type_2': mask_type_2}
        parameters.update(kwargs)
        super(ApplySecondaryMask, self).__init__(**parameters)
        mask1 = cortex.db.get_mask(surface, transform, mask_type_1).T  # in xyz
        mask2 = cortex.db.get_mask(surface, transform, mask_type_2).T  # in xyz
        self.mask = image_utils.secondary_mask(mask1, mask2, order='F')

    def run(self, x):
        if x.ndim > 1:
            x = x.reshape(-1, 1)
        return x[self.mask]


class ActivityRatio(PreprocessingStep):
    def __init__(self, *args, **kwargs):
        super(ActivityRatio, self).__init__(**kwargs)

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
    surface : str
        Subject name
    transform : str
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
    def __init__(self, surface, transform, pre_mask_name, roi_names, *args, **kwargs):
        parameters = {'surface': surface, 'transform': transform,
                      'pre_mask_name': pre_mask_name, 'roi_names': roi_names}
        parameters.update(kwargs)
        super(RoiActivity, self).__init__(**parameters)

        subj_dir = config.get_subject_directory(surface)
        pre_mask_path = op.join(subj_dir, pre_mask_name + '.nii')

        # mask in zyx
        pre_mask = nib.load(pre_mask_path).get_data().T.astype(bool)

        # returns masks in zyx
        roi_masks, roi_dict = cortex.get_roi_masks(surface, transform, roi_names)

        self.masks = dict()
        for name, mask_value in roi_dict.items():
            roi_mask = roi_masks == mask_value
            self.masks[name] = image_utils.secondary_mask(pre_mask, roi_mask)

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
    def __init__(self, subject, *args, model_name=None, **kwargs):
        parameters = {'subject': subject, 'model_name': model_name}
        parameters.update(kwargs)
        super(WMDetrend, self).__init__(**parameters)
        subj_dir = config.get_subject_directory(subject)

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


class IncrementalMeanStd(PreprocessingStep):
    """Preprocessing module that z-scores data using running mean and variance
    """
    def run(self, array):
        """Run the z-scoring on one time point and update the prior

        Parameters
        ----------
        array : numpy.ndarray
            A vector of data to be z-scored

        Returns
        -------
        The input array z-scored using the posterior mean and variance
        """
        self.update_state()
        if not getattr(self, 'data', None):
            self.array_shape = array.shape
            self.data = buffered_array.BufferedArray(array.size, dtype=array.dtype)
            self.data.append(array.ravel())
            return None, None

        self.data.append(array.ravel())

        std = np.std(self.data.get_array(), 0)
        mean = np.mean(self.data.get_array(), 0)

        return mean.reshape(self.array_shape), std.reshape(self.array_shape)

    def reset(self):
        self.data = None


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
    def __init__(self, *args, n=20, n_skip=5, **kwargs):
        parameters = {'n': n, 'n_skip': n_skip}
        parameters.update(kwargs)
        super(RunningMeanStd, self).__init__(**parameters)
        self.n = n
        self.mean = None
        self.samples = None
        self.n_skip = n_skip

    def run(self, inp, image_number=None):
        if image_number < self.n_skip:
            return np.zeros(inp.size), np.ones(inp.size)

        if self.mean is None:
            self.samples = np.empty((self.n, inp.size)) * np.nan
        else:
            self.samples[:-1, :] = self.samples[1:, :]

        self.samples[-1, :] = inp
        self.mean = np.nanmean(self.samples, 0)
        self.std = np.nanstd(self.samples, 0)
        return self.mean, self.std

    def reset(self):
        self.mean = None
        self.samples = None


class ZScore(PreprocessingStep):
    """Compute a z-scored version of an input array given precomputed means and standard deviations

    Methods
    -------
    run(inp, mean, std)
        Return the z-scored version of the data
    """
    def run(self, array, mean, std):
        if mean is None:
            zscored_array = np.zeros_like(array)
        else:
            zscored_array = np.divide(array - mean, std, where=std != 0)

        return zscored_array


class AggregateTimestampedVolumes(PreprocessingStep):
    def __init__(self, *args, active=True, buffer_size=1000, **kwargs):
        parameters = {'active': active, 'buffer_size': buffer_size}
        parameters.update(kwargs)
        super(AggregateTimestampedVolumes, self).__init__(**parameters)

        self.active = active
        self.buffer_size = buffer_size
        self.array = None
        self.times = None

    def run(self, t, array):
        self.update_state()

        if self.array is None:
            n_samples = array.size
            self.times = buffered_array.BufferedArray(size=1, buffer_size=self.buffer_size)
            self.array = buffered_array.BufferedArray(size=n_samples, buffer_size=self.buffer_size)
            self.n_samples = n_samples

        if self.active:
            self.times.append([t])
            self.array.append(array)
            times = self.times.get_array()
            array = self.array.get_array()

        else:
            times = []
            array = []

        return times, array

    def reset(self):
        self.array = None
        self.times = None


class SklearnPredictor(PreprocessingStep):
    """Run the `.predict` method of a scikit-learn predictor on incoming
    activity. Returns the predicted output.

    Parameters
    ----------
    surface : str
        subject/surface ID
    pickled_predictor : str
        filename of the pickle file containing the trained classifier

    Attributes
    ----------
    predictor : sklearn fitted learner

    Methods
    -------
    run():
        Returns the prediction
    """

    def __init__(self, surface, pickled_predictor, *args, nan_to_num=True, **kwargs):
        parameters = {'surface': surface, 'pickled_predictor': pickled_predictor,
                      'nan_to_num': nan_to_num}
        parameters.update(kwargs)
        super(SklearnPredictor, self).__init__(**parameters)
        subj_dir = config.get_subject_directory(surface)
        pickled_path = op.join(subj_dir, pickled_predictor)
        self.predictor = pickle.load(open(pickled_path, 'rb'))
        self.nan_to_num = nan_to_num

    def run(self, activity):
        activity = activity.ravel()[None]
        if self.nan_to_num:
            activity = np.nan_to_num(activity)

        prediction = self.predictor.predict(activity)[0]
        return prediction


class TopKPredictor(PreprocessingStep):
    def __init__(self, estimators, k=5):
        self.estimators = estimators
        self.k = k
       
    
    def run(self, X):
        log_prob = self.estimator.predict_log_proba(X)
        
        classes = self.estimator.classes_
        
        topklogprob = log_prob.argsort(axis=1)[:, ::-1][:, :self.k]
        return classes[topklogprob]


class SklearnMultiplePredictorsTopK(PreprocessingStep):
    """Given a list of scikit-learn predictors, runs the `.predict` method
    of each one of them on incoming activity. Returns a list with the
    predicted output.

    Parameters
    ----------
    surface : str
        subject/surface ID
    pickled_predictors : list of str
        filenames of the pickle files containing the trained classifiers
    k: int

    Attributes
    ----------
    predictors : list of sklearn fitted learner

    Methods
    -------
    run():
        Returns the prediction
    """

    def __init__(self, surface, pickled_predictors, *args, k=5, nan_to_num=True,
                 **kwargs):
        parameters = {
            'surface': surface,
            'pickled_predictors': pickled_predictors,
            'nan_to_num': nan_to_num,
            'k': k
        }
        parameters.update(kwargs)
        super(SklearnMultiplePredictorsTopK, self).__init__(**parameters)
        subj_dir = config.get_subject_directory(surface)
        if not isinstance(pickled_predictors, (list, tuple)):
            pickled_predictors = [pickled_predictors]
        pickled_paths = [op.join(subj_dir, pp) for pp in pickled_predictors]
        self.predictors = [
            pickle.load(open(pp, 'rb')) for pp in pickled_paths
        ]
        self.nan_to_num = nan_to_num
        self.k = k

    def run(self, activity):
        activity = activity.ravel()[None]
        if self.nan_to_num:
            activity = np.nan_to_num(activity)

        predictions = []
        for estimator in self.predictors:
            log_prob = estimator.predict_log_proba(activity)
            classes = estimator.classes_
            topklogprob = log_prob.argsort(axis=1)[:, ::-1][:, :self.k]
            predictions.append(classes[topklogprob][0].tolist())
        return {'pred': predictions}


class SklearnMultiplePredictors(PreprocessingStep):
    """Given a list of scikit-learn predictors, runs the `.predict` method
    of each one of them on incoming activity. Returns a list with the
    predicted output.

    Parameters
    ----------
    surface : str
        subject/surface ID
    pickled_predictors : list of str
        filenames of the pickle files containing the trained classifiers

    Attributes
    ----------
    predictors : list of sklearn fitted learner

    Methods
    -------
    run():
        Returns the prediction
    """

    def __init__(self, surface, pickled_predictors, *args, nan_to_num=True,
                 **kwargs):
        parameters = {
            'surface': surface,
            'pickled_predictors': pickled_predictors,
            'nan_to_num': nan_to_num
        }
        parameters.update(kwargs)
        super(SklearnMultiplePredictors, self).__init__(**parameters)
        subj_dir = config.get_subject_directory(surface)
        if not isinstance(pickled_predictors, (list, tuple)):
            pickled_predictors = [pickled_predictors]
        pickled_paths = [op.join(subj_dir, pp) for pp in pickled_predictors]
        self.predictors = [
            pickle.load(open(pp, 'rb')) for pp in pickled_paths
        ]
        self.nan_to_num = nan_to_num

    def run(self, activity):
        activity = activity.ravel()[None]
        if self.nan_to_num:
            activity = np.nan_to_num(activity)

        predictions = [
            predictor.predict(activity)[0] for predictor in self.predictors
        ]
        return {'pred': predictions}


class SendToDashboard(PreprocessingStep):
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
    def __init__(self, name, plot_type='marker', **kwargs):
        parameters = {'name': name, 'plot_type': plot_type}
        parameters.update(kwargs)
        super(SendToDashboard, self).__init__(**parameters)
        key_name = 'dashboard:data:' + name
        r.set(key_name, b'')
        r.set(key_name + ':type', plot_type)
        r.set(key_name + ':update', b'true')

        self.redis = r
        self.key_name = key_name

    def run(self, *args):
        data = pickle.dumps(args)
        logger.debug('SendToDashboard key_name=%s len(data)=%d', self.key_name, len(data))
        self.redis.set(self.key_name, data)
        self.redis.set(self.key_name + ':update', b'true')


class SendToPycortexViewer(PreprocessingStep):
    """Send data to the pycortex webgl viewer

    Parameters
    ----------
    name : str

    Attributes
    ----------
    redis : redis connection
    """
    def __init__(self, name, *args, **kwargs):
        parameters = {'name': name}
        parameters.update(kwargs)
        super(SendToPycortexViewer, self).__init__(**parameters)

    def run(self, data):
        r.publish("viewer", pickle.dumps(data))


class StoreToRedis(PreprocessingStep):
    """Store to redis database

    Parameters
    ----------
    key_prefix : str
        Prefix to redis key. Individual samples will be stored to the database with keys that
        append the current trial index and sample index to this prefix,
        e.g., responses:trial0000:0000

    Attributes
    ----------
    index : int
        Incrementing index
    active : bool
    """
    def __init__(self, key_prefix, *args, active=True, **kwargs):
        parameters = {'key_prefix': key_prefix, 'active': active}
        parameters.update(kwargs)
        super(StoreToRedis, self).__init__(**parameters)
        self.key_prefix = key_prefix
        self.index = 0
        self.active = active

    def update_state(self):
        super(StoreToRedis, self).update_state()

        trial = r.get('experiment:trial:current')
        if trial is None:
            key = f'{self.key_prefix}:pretrial'

        else:
            trial = pickle.loads(trial)
            trial_index = trial['index']

            if trial_index > 9999:
                warnings.warn('Trial index overflow (max 9999 trials). Sorting trials as strings will fail.')

            key = f'{self.key_prefix}:trial{trial_index:04}'

        self.key = key

    def run(self, *args):
        self.update_state()

        if self.active:
            key = f'{self.key}:{self.index:04}'
            r.set(key, pickle.dumps(args))
            self.index += 1

            return key

    def reset(self):
        # XXX: should we reset the index?
        pass


class PublishToRedis(PreprocessingStep):
    """Publish using redis pubsub channel

    Parameters
    ----------
    topic : str
    """
    def __init__(self, topic, *args, **kwargs):
        parameters = {'topic': topic}
        parameters.update(kwargs)
        super(PublishToRedis, self).__init__(**parameters)
        self.topic = topic

    def run(self, data):
        r.publish(self.topic, pickle.dumps(data))


class Dictionary(PreprocessingStep):
    """A python-style dict as a preprocessing step

    Parameters
    ----------
    dictionary : dict
    """
    def __init__(self, dictionary, *args, decode_key=None, **kwargs):
        parameters = {'dictionary': dictionary, 'decode_key': decode_key}
        kwargs.update(parameters)
        super(Dictionary, self).__init__(**parameters)
        self.dictionary = dictionary
        self.decode_key = decode_key

    def run(self, key):
        if self.decode_key:
            key = key.decode(self.decode_key)

        value = self.dictionary[key]
        logger.debug('Dictionary, key=%s, value=%s', key, value)
        return value


import requests
import json

class SklearnPredictorToAWS(PreprocessingStep):                                                
    """Run the `.predict` method of a scikit-learn predictor on incoming                       
    activity. Returns the predicted output.                                                    
                                                                                               
    Parameters                                                                                 
    ----------                                                                                 
    surface : str                                                                              
        subject/surface ID                                                                     
    pickled_predictor : str                                                                    
        filename of the pickle file containing the trained classifier                          
                                                                                               
    Attributes                                                                                 
    ----------                                                                                 
    predictor : sklearn fitted learner                                                         
                                                                                               
    Methods                                                                                    
    -------                                                                                    
    run():                                                                                     
        Returns the prediction                                                                 
    """                                                                                        
                                                                                               
    def __init__(self, surface,  pickled_predictors, aws_address, nan_to_num=True, **kwargs):  
        parameters = {'surface': surface, 'pickled_predictors':                                
                      pickled_predictors,                                                      
                      'nan_to_num': nan_to_num, 'aws_address': aws_address}                    
        parameters.update(kwargs)                                                              
        super(SklearnPredictorToAWS, self).__init__(**parameters)                              
        subj_dir = config.get_subject_directory(surface)                                       
        pickled_paths = {name: op.join(subj_dir, pickled_predictor) for                        
                         name, pickled_predictor in pickled_predictors.items()}                
        self.predictors = {name: pickle.load(open(pickled_path, 'rb')) for                     
                           name, pickled_path in pickled_paths.items()}                        
        self.nan_to_num = nan_to_num                                                           
        self.aws_address = aws_address                                                         
                                                                                               
    def run(self, activity, experiment_info):                                                                   
        activity = activity.ravel()[None]                                                      
        if self.nan_to_num:                                                                    
            activity = np.nan_to_num(activity)                                                 
                                                                                               
        probabilities = {name:predictor.predict_proba(activity)[0]
								for name, predictor in self.predictors.items()}
		
        concept_classes = self.predictors["concepts"].classes_                                              
        sizes_list = [dict(name=f"concept_{class_name}", size=probability)                     
                         for class_name, probability in zip(concept_classes,                   
                                                            probabilities["concepts"])]
        category_classes = self.predictors["categories"].classes_
        sizes_list.extend([dict(name=f"category_{class_name}", size=probability)                     
                         for class_name, probability in zip(category_classes,                   
                                                            probabilities["categories"])])

        json_sizes_list = json.dumps(sizes_list)
        data = dict(sizes=json_sizes_list)
        if experiment_info is not None:
            data['cue'] = experiment_info['cur_cue']

        requests.post(self.aws_address, data=data)                      
                                                                                               

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

class UploadFlatmap(PreprocessingStep):
    """Computes a flatmap and pushes it to an http address

    Parameters
    ----------
    surface: str
        subject/surface-ID

    transform: str
        pycortex transform name

    height: int, default 1024
        height in pixels of the flatmap

    address: str
        location to http POST the flatmap to under key 'flatmap'

    """

    def __init__(self, surface, transform, address, height=1024, vmin=None,
                 vmax=None, cmap=None, **kwargs):
        parameters = dict(surface=surface, transform=transform,
                          address=address, height=height)
        parameters = {**kwargs, **parameters}
        super(UploadFlatmap, self).__init__(**parameters)

        self.surface = surface
        self.transform = transform
        self.address = address
        self.height = height
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap

    def run(self, activity):
        
        volume = cortex.Volume(activity, self.surface, self.transform,
                              cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)

        buf = io.BytesIO()

        fig = cortex.quickflat.make_figure(volume, height=self.height)

        plt.savefig(buf, format="PNG")

        buf.seek(0)
        content = buf.read()

        requests.post(self.address, data={"flatmap.png": content})


class SimulateDecodingProba(PreprocessingStep):
    """Simulate decoding probabilities by randomly sampling from a dirichlet
    distribution

    Parameters
    ----------
    n_classes: int
    """

    def __init__(self, n_classes, **kwargs):
        parameters = dict(n_classes=n_classes)
        parameters = {**kwargs, **parameters}
        super(SimulateDecodingProba, self).__init__(**parameters)
        self.n_classes = n_classes

    def run(self, activity):
        return np.random.dirichlet([1] * self.n_classes)




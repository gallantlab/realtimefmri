#!/usr/bin/env python
import os
import subprocess
import cPickle
from glob import glob
from itertools import izip
import time
import yaml
import json
import warnings
from uuid import uuid4

import logging
logger = logging.getLogger('preprocess.ing')
logger.setLevel(logging.DEBUG)

import numpy as np
import zmq

from nibabel import save as nbsave, load as nbload
from nibabel.nifti1 import Nifti1Image
import cortex

from image_utils import transform, mosaic_to_volume
from .utils import get_database_directory, get_recording_directory, get_subject_directory

db_dir = get_database_directory()
rec_dir = get_recording_directory()

class Preprocessor(object):
	'''
	This class loads the preprocessing pipeline from the configuration
	file, initializes the classes for each step, and runs the main loop
	that receives incoming images from the data collector.
	'''
	def __init__(self, preproc_config, **kwargs):
		super(Preprocessor, self).__init__()

		# initialize input and output sockets
		context = zmq.Context()
		self.input_socket = context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:5556')
		self.input_socket.setsockopt(zmq.SUBSCRIBE, 'image')

		self.output_socket = context.socket(zmq.PUB)
		self.output_socket.bind('tcp://*:5557')

		self.active = False

		# load the pipeline from pipelines.conf
		with open(os.path.join(db_dir, preproc_config+'.conf'), 'r') as f:
			yaml_contents = yaml.load(f)
			subject = yaml_contents['subject']
			pipeline = yaml_contents['pipeline']

		for step in pipeline:
			logger.info('initializing %s' % step['name'])
			step['instance'].__init__(**step.get('kwargs', {}))

		self.pipeline = pipeline

	def run(self):
		self.active = True
		logger.info('ready')
		while self.active:
			message = self.input_socket.recv()
			data = message[6:]
			logger.info('received image data of length %u' % len(data))
			outp = self.process(data)
			time.sleep(0.1)

	def process(self, raw_image_binary):
		data_dict = {
			'raw_image_binary': raw_image_binary,
		}

		for step in self.pipeline:
			logger.debug('running %s' % step['name'])
			args = [data_dict[i] for i in step['input']]
			outp = step['instance'].run(*args)
			if not isinstance(outp, (list, tuple)):
				outp = [outp]
			d = dict(zip(step.get('output', []), outp))
			data_dict.update(d)
			for topic in step.get('send', []):
				logger.info('sending %s' % topic)
				if isinstance(d[topic], dict):
					self.output_socket.send(topic+' '+json.dumps(d[topic]))
				elif isinstance(d[topic], (np.ndarray)):
					self.output_socket.send(topic+' '+d[topic].astype(np.float32).tostring())

		return data_dict

class PreprocessingStep(object):
	def __init__(self):
		pass
	def run(self):
		raise NotImplementedError

class Debug(PreprocessingStep):
	def run(self, val):
		logger.info('debugging')
		if isinstance(val, np.ndarray):
			val = str(val[:10])
		logger.info(val)

class RawToNifti(PreprocessingStep):
	'''
	takes data_dict containing raw_image_binary and adds 
	'''
	def __init__(self, subject=None, reference_name='funcref.nii'):
		if not subject is None:
			funcref_path = os.path.join(get_subject_directory(subject), reference_name)
			funcref = nbload(funcref_path)
			self.affine = funcref.affine[:]
		else:
			warnings.warn('No subject provided. Set affine attribute manually before calling run.')

	def run(self, inp):
		'''
		pixel_image is a binary string loaded directly from the .PixelData file
		saved on the scanner console

		returns a nifti1 image of the same data in xyz
		'''
		# siements mosaic format is strange
		mosaic = np.fromstring(inp, dtype=np.uint16).reshape(600,600, order='C')
		# axes 0 and 1 must be swapped because mosaic is PLS and we need LPS voxel data
		# (affine values are -/-/+ for dimensions 1-3, yielding RAS)
		# we want the voxel data orientation to match that of the functional reference, gm, and wm masks
		volume = mosaic_to_volume(mosaic).swapaxes(0,1)[...,:30]
		return Nifti1Image(volume, self.affine)

class SaveNifti(PreprocessingStep):
	def __init__(self, save_directory=None, path_format='volume_%4.4u.nii'):
		if save_directory is None:
			save_directory = str(uuid4())
		self.save_directory = os.path.join(rec_dir, save_directory)
		self.path_format = path_format
		self._i = 0
		try:
			os.mkdir(self.save_directory)
		except OSError:
			self._i = self._infer_i()
			warnings.warn('''Save directory already exists. Beginning file numbering with %u''' % self._i)
	def _infer_i(self):
		from re import compile as re_compile
		pattern = re_compile("\%[0-9]*\.?[0-9]*[uifd]")
		match = pattern.split(self.path_format)
		glob_pattern = '*'.join(match)
		fpaths = glob(os.path.join(self.save_directory, glob_pattern))

		i_pattern = re_compile('(?<={})[0-9]*(?={})'.format(*match))
		try:
			max_i = max([int(i_pattern.findall(i)[0]) for i in fpaths])
			i = max_i + 1
		except ValueError:
			i = 0
		return i

	def run(self, inp):
		fpath = self.path_format % self._i
		nbsave(inp, os.path.join(self.save_directory, fpath))
		self._i += 1

class MotionCorrect(PreprocessingStep):
	def __init__(self, subject=None, reference_name='funcref.nii'):
		if (subject is not None):
			self.reference_path = os.path.join(get_subject_directory(subject), reference_name)
			self.reference_affine = nbload(self.reference_path).affine
		else:
			warnings.warn('''Provide path to reference volume before calling run.''')

	def run(self, input_volume):
		assert np.allclose(input_volume.affine, self.reference_affine)
		return transform(input_volume, self.reference_path)

class ApplyMask(PreprocessingStep):
	'''
	Loads a mask from the realtimefmri database
	Mask should be in xyz format to match data.
	Mask is applied after transposing mask and data to zyx
	to match the wm detrend training
	'''
	def __init__(self, subject=None, mask_name=None):
		if (subject is not None) and (mask_name is not None):
			subj_dir = get_subject_directory(subject)
			mask_path = os.path.join(subj_dir, mask_name+'.nii')
			self.load_mask(mask_path)
		else:
			warnings.warn('''Load mask manually before calling run.''')

	def load_mask(self, mask_path):
			mask_nifti1 = nbload(mask_path)
			self.mask_affine = mask_nifti1.affine
			self.mask = mask_nifti1.get_data().astype(bool)

	def run(self, volume):
		assert np.allclose(volume.affine, self.mask_affine)
		return volume.get_data().T[self.mask.T]

class ApplyMask2(PreprocessingStep):
	def __init__(self, subject, mask1_name, mask2_name):
		'''
		Input:
		-----
		mask1_path: path to a boolean mask in xyz format
		mask2_path: path to a boolean mask in xyz format

		Initialization will generate a boolean vector that selects elements 
		from the vector output of mask1 applied to a volume that are also in
		mask2.

		Example:
		-----
		Given a 3d array, X and two 3d masks, mask1 and mask2
		X.T[mask1.T] = x
		X.T[mask1.T & mask2.T] = y
		x[new_mask] = y
		'''
		subj_dir = get_subject_directory(subject)
		mask1_path = os.path.join(subj_dir, mask1_name+'.nii')
		mask2_path = os.path.join(subj_dir, mask2_name+'.nii')

		mask1 = nbload(mask1_path).get_data().astype(bool) # in xyz
		mask2 = nbload(mask2_path).get_data().astype(bool) # in xyz		
		mask1_flat = mask1.flatten(order='F')
		mask2_flat = mask2.flatten(order='F')

		masks = np.c_[mask1_flat, mask2_flat]
		masks = masks[mask1_flat,:]
		self.mask = masks[:,1].astype(bool)

	def run(self, x):
		if x.ndim>1:
			x = x.reshape(-1,1)
		return x[self.mask]

class ActivityRatio(PreprocessingStep):
	def run(self, x1, x2):
		if isinstance(x1, np.ndarray):
			x1 = np.nanmean(x1)
		if isinstance(x2, np.ndarray):
			x2 = np.nanmean(x2)

		return x1/(x1+x2)

class RoiActivity(PreprocessingStep):
	def __init__(self, subject, xfm_name, pre_mask_name, roi_names):
		subj_dir = get_subject_directory(subject)
		pre_mask_path = os.path.join(subj_dir, pre_mask_name+'.nii')
		
		# mask in zyx
		pre_mask = nbload(pre_mask_path).get_data().T
		pre_mask_ix = pre_mask.flatten().nonzero()[0]

		# returns masks in zyx
		roi_masks, roi_dict = cortex.get_roi_masks(subject, xfm_name, roi_names)
		self.masks = dict()
		for name, mask_value in roi_dict.iteritems():
			mask = roi_masks==mask_value
			mask_overlap = np.logical_and(pre_mask, mask).flatten().nonzero()[0]
			self.masks[name] = np.asarray([i for i,j in enumerate(pre_mask_ix) if j in mask_overlap])

	def run(self, activity):
		if activity.ndim>1:
			activity = activity.reshape(-1,1)
		roi_activities = dict()
		for name, mask in self.masks.iteritems():
			roi_activities[name] = float(activity[mask].mean())
		return roi_activities

class WMDetrend(PreprocessingStep):
	'''
	Stores the white matter mask in functional reference space
	when a new image comes in, motion corrects it to reference image
	then applies wm mask
	'''
	def __init__(self, subject, model_name=None):
		'''
		This should set up the class instance to be ready to take an image input
		and output the detrended gray matter activation

		To do this, it needs:
			wm_mask_funcref: white matter masks in functional reference space
			gm_mask_funcref: gray matter masks in functional reference space
			funcref_nifti1: the functional reference image
			input_affine: affine transform for the input images
				since we'll be dealing with raw pixel data, we need to
				have a predetermined image orientation
		'''
		self.subj_dir = get_subject_directory(subject)
		self.subject = subject

		self.funcref_nifti1 = nbload(os.path.join(self.subj_dir, 'funcref.nii'))

		try:
			model_path = os.path.join(self.subj_dir, 'model-%s.pkl'%model_name)
			pca_path = os.path.join(self.subj_dir, 'pca-%s.pkl'%model_name)

			with open(model_path, 'r') as f:
				model = cPickle.load(f)
			self.model = model

			with open(pca_path, 'r') as f:
				pca = cPickle.load(f)
			self.pca = pca
		except IOError:
			warnings.warn('''Could not load...\n\tModel from %s\nand\n\tPCA from %s. Load them manually before running.''' % (model_path, pca_path))
		
	def train(self, gm_train, wm_train, n_wm_pcs=10):
		from sklearn.linear_model import LinearRegression
		from sklearn.decomposition import PCA
		
		n_trials, n_wm_voxels = wm_train.shape
		_, n_gm_voxels = gm_train.shape

		pca = PCA(n_components=n_wm_pcs)
		wm_train_pcs = pca.fit_transform(wm_train)
		
		model = LinearRegression()
		model.fit(wm_train_pcs, gm_train)

		return model, pca

	def run(self, wm_activity, gm_activity):
		wm_activity_pcs = self.pca.transform(wm_activity.reshape(1,-1)).reshape(1,-1)
		gm_trend = self.model.predict(wm_activity_pcs)
		return gm_activity - gm_trend

def compute_raw2var(raw1,raw2,*args):
	'''Use the raw moments to compute the 2nd central moment
	VAR(X) = E[X^2] - E[X]^2
	'''
	return raw2 - raw1**2
def compute_raw2skew(raw1,raw2,raw3,*args):
	'''Use the raw moments to compute the 3rd standardized moment
	Skew(X) = (E[X^3] - 3*E[X]*E[X^2] + 2*E[X]^3)/VAR(X)^(3/2)
	'''
	# get central moments
	cm2 = raw2 - raw1**2
	cm3 = raw3 - 3*raw1*raw2 + 2*raw1**3
	# get standardized 3rd moment
	sm3 = cm3/cm2**1.5
	return sm3

def compute_raw2kurt(raw1,raw2,raw3,raw4, *args):
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
	for raw_moment in izip(*node_raw_moments):
		moment = np.sum(raw_moment, 0)/nsamples
		mean_moments.append(moment)

	emean = mean_moments[0]
	evar = compute_raw2var(*mean_moments)
	eskew = compute_raw2skew(*mean_moments)
	ekurt = compute_raw2kurt(*mean_moments)
	return emean, evar, eskew, ekurt

class OnlineMoments(PreprocessingStep):
	'''Compute 1-Nth raw moments online

	For the Ith moment: E[X^i] = (1/n)*\Sum(X^i)
	This function only stores \Sum(X^i) and keeps
	track of the number of observations.
	'''
	def __init__(self, order=4):
		self.n = 0.0
		self.order = order
		self.all_raw_moments = [0.0]*self.order
		for odx in xrange(self.order):
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

		for odx in xrange(self.order):
			name = 'rawmnt%i'%(odx+1)
			self.all_raw_moments[odx] = self.all_raw_moments[odx] + x**(odx+1)
			self.__setattr__(name, self.all_raw_moments[odx])

	def get_statistics(self):
		'''Return the 1,2,3,4-moment estimates'''
		return convert_parallel2moments([self.get_raw_moments()[:4]], self.n) # mean,var,skew,kurt

	def get_raw_moments(self):
		return self.all_raw_moments

	def get_norm_raw_moments(self):
		return map(lambda x: x/float(self.n), self.all_raw_moments)

	def run(self, inp):
		self.update(inp)
		return self.get_statistics()[:2]

class RunningMeanStd(PreprocessingStep):
	def __init__(self, n=20):
		self.n = n
		self.mean = None
	def run(self, inp):
		if self.mean is None:
			self.samples = np.empty((self.n, inp.size))*np.nan
		else:
			self.samples[:-1,:] = self.samples[1:,:]
		self.samples[-1,:] = inp
		self.mean = np.nanmean(self.samples, 0)
		self.std = np.nanstd(self.samples, 0)
		return self.mean, self.std
	
class VoxelZScore(PreprocessingStep):
	def __init__(self):
		self.mean = None
		self.std = None
	def zscore(self, data):
		return (data-self.mean)/self.std
	def run(self, inp, mean=None, std=None):
		if not mean is None:
			self.mean = mean
		if not std is None:
			self.std = std

		if self.mean is None:
			z = inp
		else:
			if (self.std==0).all():
				z = np.zeros_like(inp)
			else:
				z = self.zscore(inp)
		return z
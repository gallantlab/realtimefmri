#!/usr/bin/env python
import os
import subprocess
import cPickle
from glob import glob
from itertools import izip
import time
import yaml
import argparse
import warnings

import logging

import numpy as np
import zmq
from matplotlib import pyplot as plt

from nibabel import save as nbsave, load as nbload
from nibabel.nifti1 import Nifti1Image
import cortex

from image_utils import transform, mosaic_to_volume
import utils

db_dir = utils.get_database_directory()

# initialize root logger, assigning file handler to output messages to log file
logging.basicConfig(level=logging.DEBUG,
	format='%(asctime)-12s %(name)-16s %(levelname)-8s %(message)s',
	filename=os.path.join(utils.get_log_directory(), '%s_preprocessing.log'%time.strftime('%Y%m%d')),
	filemode='a')

# add logger, add stream handler to output to console
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

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
			d = dict(zip(step['output'], outp))
			data_dict.update(d)
			for send in step.get('send', []):
				logger.debug('sending %s' % send)
				self.output_socket.send('%s %s' % (send, d[send].astype(np.float32).tostring()))

		return data_dict

class Debug(object):
	def run(self, data_dict):
		logger.debug(data_dict.keys())
		return {}

class RawToNifti(object):
	'''
	takes data_dict containing raw_image_binary and adds 
	'''
	def __init__(self, subject=None):
		if not subject is None:
			funcref_path = os.path.join(utils.get_subject_directory(subject), 'funcref.nii')
			funcref = nbload(funcref_path)
			self.affine = funcref.affine[:]
		else:
			warnings.warn('No subject provided. Set affine attribute manually before calling run.')

	def run(self, inp):
		'''
			pixel_image is a binary string loaded directly from the .PixelData file
			saved on the scanner console

			returns a nifti1 image of the same data
		'''
		# siements mosaic format is strange
		mosaic = np.fromstring(inp, dtype=np.uint16).reshape(600,600, order='C')
		# axes 0 and 1 must be swapped because mosaic is PLS and we need LPS voxel data
		# (affine values are -/-/+ for dimensions 1-3, yielding RAS)
		# we want the voxel data orientation to match that of the functional reference, gm, and wm masks
		volume = mosaic_to_volume(mosaic).swapaxes(0,1)[...,:30]
		return Nifti1Image(volume, self.affine)

class MotionCorrect(object):
	def __init__(self, subject=None, reference_name='funcref.nii'):
		if (subject is not None):
			self.reference_path = os.path.join(utils.get_subject_directory(subject), reference_name)
			self.reference_affine = nbload(self.reference_path).affine
		else:
			warnings.warn('''Provide path to reference volume before calling run.''')

	def run(self, input_volume):
		assert np.allclose(input_volume.affine, self.reference_affine)
		return transform(input_volume, self.reference_path)

class ApplyMask(object):
	def __init__(self, subject=None, mask_name=None):
		if (subject is not None) and (mask_name is not None):
			subj_dir = utils.get_subject_directory(subject)
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

class RoiActivity(object):
  def __init__(self, subject, xfm_name, pre_mask_name, roi_names):
    subj_dir = utils.get_subject_directory(subject)
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
      print self.masks[name].max()

  def run(self, activity):
    if activity.ndim>1:
      activity = activity.reshape(-1,1)
    roi_activities = dict()
    for name, mask in self.masks.iteritems():
      roi_activities[name] = activity[mask]
    return roi_activities[self.masks.keys()[0]].mean()

class WMDetrend(object):
	'''
	Stores the white matter mask in functional reference space
	when a new image comes in, motion corrects it to reference image
	then applies wm mask
	'''
	def __init__(self, subject, model_name):
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
		self.subj_dir = utils.get_subject_directory(subject)
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

class OnlineMoments(object):
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

class RunningMeanStd(object):
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
	
class VoxelZScore(object):
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
			return inp
		else:
			return self.zscore(inp)

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Preprocess data')
	parser.add_argument('config',
		action='store',
		nargs='?',
		default='preproc-01',
		help='Name of configuration file')
	args = parser.parse_args()
	logger.info('Loading preprocessing pipeline from %s' % args.config)

	preproc = Preprocessor(args.config)
	preproc.run()
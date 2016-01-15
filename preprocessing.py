#!/usr/bin/env python
import os
import subprocess
import cPickle
from glob import glob
import time
import yaml

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
subj_dir = utils.get_subject_directory('S1')

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

input_affine = np.array([
		[-2.24000001, 0., 0., 108.97336578],
		[0., -2.24000001, 0., 131.32203674],
		[0., 0., 4.12999725, -59.81945419],
		[0., 0., 0., 1.]
	])

class Preprocessor(object):
	def __init__(self, preproc_config, **kwargs):
		super(Preprocessor, self).__init__()
		context = zmq.Context()
		self.input_socket = context.socket(zmq.SUB)
		self.input_socket.connect('tcp://localhost:5556')
		self.input_socket.setsockopt(zmq.SUBSCRIBE, 'image')

		self.output_socket = context.socket(zmq.PUB)
		self.output_socket.bind('tcp://*:5557')

		self.active = False

		# load the pipeline fron pipelines.conf
		with open(os.path.join(db_dir, preproc_config+'.conf'), 'r') as f:
			self.preproc_pipeline = yaml.load(f)['preprocessing']

		for step in self.preproc_pipeline:
			logger.info('initializing %s' % step['name'])
			step['instance'].__init__()

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

		for step in self.preproc_pipeline:
			logger.debug('running %s' % step['name'])
			args = [data_dict[i] for i in step['input']]
			d = step['instance'].run(*args)
			data_dict.update(d)
			for out in step.get('output', []):
				logger.debug('outputting %s' % out)
				self.output_socket.send('%s %s' % (out, d[out].astype(np.float32).tostring()))

		return data_dict

class Debug(object):
	def run(self, data_dict):
		logger.debug(data_dict.keys())
		return {}

class RawToVolume(object):
	'''
	takes data_dict containing raw_image_binary and adds 
	'''
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
		volume = mosaic_to_volume(mosaic).swapaxes(0,1)[..., 2:26]
		return { 'raw_image_volume': volume }

class Average(object):
	def __init__(self):
		self.average_image = None
		self.n_images = 0

	def run(self, input_image):

		if self.n_images==0:
			self.average_image = input_image
			return self.average_image
			
		return self.average_image

class WMDetrend(object):
	'''
	Stores the white matter mask in functional reference space
	when a new image comes in, motion corrects it to reference image
	then applies wm mask
	'''
	def __init__(self, subject='S1', model_name='20160112_1804'):
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
		self.masks = self.get_masks()

		self.funcref_nifti1 = nbload(os.path.join(self.subj_dir, 'funcref.nii'))

		self.input_affine = input_affine

		model_path = os.path.join(self.subj_dir, 'model-%s.pkl'%model_name)
		with open(model_path, 'r') as f:
			model = cPickle.load(f)
		self.model = model

		pca_path = os.path.join(self.subj_dir, 'pca-%s.pkl'%model_name)
		with open(pca_path, 'r') as f:
			pca = cPickle.load(f)
		self.pca = pca

	def get_masks(self):
		masks = dict()
		try:
			wm_mask_funcref = nbload(os.path.join(self.subj_dir, 'wm_mask_funcref.nii'))
			masks['wm'] = wm_mask_funcref.get_data().astype(bool)
			logger.debug('loaded white matter mask')
		except OSError:
			logger.debug('white matter mask not found`')

		try:
			gm_mask_funcref = nbload(os.path.join(self.subj_dir, 'gm_mask_funcref.nii'))
			masks['gm'] = gm_mask_funcref.get_data().astype(bool)
			logger.debug('loaded gray matter mask')
		except OSError:
			logger.debug('gray matter mask not found')

		return masks
		
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

	def get_activity_in_masks(self, input_voxeldata):
		input_nifti1 = Nifti1Image(input_voxeldata, self.input_affine)
		input_funcref_nifti1 = transform(input_nifti1, self.funcref_nifti1.get_filename())

		return {
			'wm': input_funcref_nifti1.get_data().T[self.masks['wm'].T],
			'gm': input_funcref_nifti1.get_data().T[self.masks['gm'].T]
		}

	def detrend(self, input_voxeldata):
		activity = self.get_activity_in_masks(input_voxeldata)
	
		wm_activity_pcs = self.pca.transform(activity['wm'].reshape(1,-1)).reshape(1,-1)
		gm_trend = self.model.predict(wm_activity_pcs)
		
		return activity['gm']-gm_trend

	def run(self, inp):
		gm_detrend = self.detrend(inp)
		return { 'gm_detrend': gm_detrend }

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
		return { 'running_mean': self.mean, 'running_std': self.std}
		
class VoxelZScore(object):
	def __init__(self):
		tmp = np.load(os.path.join(subj_dir, 'gm_zscore.npz'))
		self.mean = tmp['mean']
		self.std = tmp['std']
	def zscore(self, data):
		return (data-self.mean)/self.std
	def run(self, inp, mean=None, std=None):
		if not mean is None:
			self.mean = mean
		if not std is None:
			self.std = std
		return { 'gm_zscore': self.zscore(inp)}

if __name__=='__main__':
	preproc = Preprocessor('preproc-01')
	preproc.run()
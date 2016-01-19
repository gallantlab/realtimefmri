#!/usr/bin/env python
import os
import subprocess
import cPickle
from glob import glob
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
			if not isinstance(outp, list):
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

class ApplyMask(object):
	def __init__(self, subject=None, mask_name=None):
		if (subject is not None) and (mask_name is not None):
			subj_dir = utils.get_subject_directory(subject)
			mask_path = os.path.join(subj_dir, mask_name+'.nii')
			self.load_mask(mask_path)
		else:
			warnings.warn('''Set "subject" and "mask_name" attributes manually before calling run.''')

	def load_mask(self, mask_path):
			mask_nifti1 = nbload(mask_path)
			self.mask_affine = mask_nifti1.affine
			self.mask = mask_nifti1.get_data().astype(bool)

	def run(self, volume):
		assert np.allclose(volume.affine, self.mask_affine)
		return volume.get_data()[self.mask]
		

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
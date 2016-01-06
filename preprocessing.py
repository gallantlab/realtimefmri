#!/usr/bin/env python
import numpy as np
import zmq
import time
import yaml
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import cPickle
from glob import glob
from uuid import uuid4
import os
import subprocess

from nibabel import save as nbsave, load as nbload
from nibabel.nifti1 import Nifti1Image

from matplotlib import pyplot as plt

import cortex

from image_utils import transform, mosaic_to_volume
import utils

db_dir = utils.get_database_directory()
subj_dir = utils.get_subject_directory('S1')

input_affine = np.array([
		[-2.24000001, 0., 0., 108.97336578],
		[0., -2.24000001, 0., 131.32203674],
		[0., 0., 4.12999725, -59.81945419],
		[0., 0., 0., 1.]
	])



class Preprocessor(object):
	def __init__(self, pipeline_name='01', **kwargs):
		super(Preprocessor, self).__init__()
		context = zmq.Context()
		self.image_subscribe = context.socket(zmq.SUB)
		self.image_subscribe.connect('tcp://localhost:5556')
		self.image_subscribe.setsockopt(zmq.SUBSCRIBE, 'image')
		self.active = False

		# load the pipeline fron pipelines.conf
		with open(os.path.join(db_dir, 'pipelines.conf'), 'r') as f:
			self.pipeline = yaml.load(f)['pipeline-'+pipeline_name]

		for step in self.pipeline:
			logger.debug('initializing %s' % step['name'])
			step['instance'].__init__()

	def run(self):
		self.active = True
		logger.debug('running')
		while self.active:
			message = self.image_subscribe.recv()
			data = message.strip('image ')
			outp = self.process(data)
			time.sleep(0.1)

	def process(self, raw_image_binary):
		data_dict = {
			'raw_image_binary': raw_image_binary,
		}

		for step in self.pipeline:
			logger.debug('running %s' % step['name'])
			output_dict = step['instance'].run(data_dict)
			data_dict.update(output_dict)

		return data_dict

class Debug(object):
	def run(self, inp):
		logger.debug(inp.keys())
		return {}

class RawToVolume(object):
	'''
	takes data_dict containing raw_image_binary and adds 
	'''
	def run(self, data_dict):
		'''
			pixel_image is a binary string loaded directly from the .PixelData file
			saved on the scanner console

			returns a nifti1 image of the same data
		'''
		mosaic = np.fromstring(data_dict['raw_image_binary'], dtype=np.uint16).reshape(600,600)
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
	def __init__(self, subject='S1'):
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

		try:
			model_paths = glob(os.path.join(self.subj_dir, 'model*.pkl'))
			with open(model_paths[0], 'r') as f:
				model = cPickle.load(f)
			self.model = model
		except IndexError:
			pass

		try:
			pca_paths = glob(os.path.join(self.subj_dir, 'pca*.pkl'))
			with open(pca_paths[0], 'r') as f:
				pca = cPickle.load(f)
			self.pca = pca
		except IndexError:
			pass

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

	def run(self, data_dict):
		gm_detrend = self.detrend(data_dict['raw_image_volume'])
		return { 'gm_detrend': gm_detrend }


class Visualizer(object):
	def __init__(self):
		self.fig, self.ax = plt.subplots()
	def run(self, image):
		self.ax.cla()
		self.ax.pcolormesh(image.get_data()[..., 10])
		self.fig.savefig('vis.png')
		return {}

if __name__=='__main__':
	preproc = Preprocessor('01')
	preproc.run()
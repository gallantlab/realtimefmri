import os
from subprocess import call

import numpy as np
from uuid import uuid4
from nibabel import load as nbload, save as nbsave

from utils import generate_command

class PixelToNifti1(object):
	'''
	takes data_dict containing raw_image_binary and adds 
	'''
	def __init__(self):
		self.transform = np.eye(4)

	def mosaic_to_volume(self, mosaic, nrows=6, ncols=6):
		nrows = 6
		volume = np.empty((100,100,36))
		for i in xrange(nrows):
			volume[:,:,i*ncols:(i+1)*ncols] = mosaic[i*100:(i+1)*100, :].reshape(100,100,ncols,order='F')
		return volume

	def fromstring(self, img_bin):
		return np.fromstring(img_bin, dtype=np.uint16).reshape(600,600)

	def run(self, data_dict):
		'''
			pixel_image is a binary string loaded directly from the .PixelData file
			saved on the scanner console

			returns a nifti1 image of the same data
		'''
		pixel_bin = 'raw'
		pixel_img = self.fromstring(pixel_bin)
		volume_img = self.mosaic_to_volume(pixel_img)
		return { 'raw_image_nifti': Nifti1Image(volume_img, self.transform) }

def transform(inp, base, verbose=False, cleanup=True):
	if type(base)==str:
		base_path = base
	else:
		base_path = str(uuid4())+'.nii'
		nbsave(inp, base_path)

	if type(inp)==str:
		inp_path = inp
	else:	
		inp_path = str(uuid4())+'.nii'
		nbsave(inp, inp_path)

	out_path = str(uuid4())+'.nii'
	params = [
		{
			'name': 'input file path',
			'position': 'last',
			'value': inp_path
		},
		{
			'name': 'reference path',
			'flag': 'base',
			'value': base_path
		},
		{
			'name': 'output file path',
			'flag': 'prefix',
			'value': out_path
		}
	]
	cmd = generate_command('3dvolreg', params)

	ret = call(cmd)
	if verbose:
		print ' '.join(cmd)
		print 'ret = %i' % ret
		
	if ret==0:
		out_img = nbload(out_path)
		out_img.get_data()

		if cleanup:
			os.remove(out_path)
			if inp is not inp_path:
				os.remove(inp_path)
			if base is not base_path:
				os.remove(base_path)
	
		return out_img

	elif ret==1:
		print ' '.join(cmd)

def mosaic_to_volume(mosaic, nrows=6, ncols=6):
	volume = np.empty((100,100, nrows*ncols))
	for i in xrange(nrows):
		volume[:,:,i*ncols:(i+1)*ncols] = mosaic[i*100:(i+1)*100, :].reshape(100,100,ncols,order='F')
	return volume

def plot_volume(volume):
	nslices = volume.shape[2]
	nrows = ncols = np.ceil(nslices**0.5).astype(int)
	
	fig, ax = plt.subplots(nrows, ncols)
	for i in xrange(volume.shape[2]):
		ax[divmod(i,ncols)].pcolormesh(volume[:,:,i], cmap='gray');


def read_pixel_data(img_fpath):
	with open(img_fpath, 'r') as f:
		img = np.fromstring(f.read(), dtype=np.uint16).reshape(600,600)
	return img
import unittest
from nibabel.nifti1 import Nifti1Image
from nibabel import load as nbload, save as nbsave
from realtimefmri.data_collection import MonitorDirectory, get_example_data_directory
from realtimefmri.preprocessing import RawToVolume, WMDetrend, VoxelZScore, RunningMeanStd
from realtimefmri.image_utils import transform, load_afni_xfm
from realtimefmri.utils import get_test_data_directory
import logging
FORMAT = '%(levelname)s: %(name)s %(funcName)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger(__name__)
import os
from glob import iglob

import time

import numpy as np

test_data_directory = get_test_data_directory()

class MonitorDirectoryTests(unittest.TestCase):
	
	def test_get_directory_contents(self):
		'''
		make sure m.get_directory_contents contains all files with image_extension
		'''
		try:
			os.mkdir('tmp')

			# create dummy file
			open('tmp/tmp.tmp', 'w').close()

			m = MonitorDirectory('tmp', image_extension='.tmp')
			directory_contents = m.get_directory_contents()
			for i in directory_contents:
				logger.info('directory content %s' % i)

			self.assertEquals(directory_contents, set(['tmp.tmp']))
		except:
			pass
		finally:
			[os.remove(i) for i in iglob('tmp/*.tmp')]
			os.rmdir('tmp')

	def test_add_image_to_shared_folder(self):
		'''
		monitor file contents of a directory add a file and assert that in a call to new_image_paths = m.get_new_image_paths() new_image_paths contains the new file
		'''
		os.mkdir('tmp')
		m = MonitorDirectory('tmp', image_extension='.tmp')

		# create dummy file
		open('tmp/tmp.tmp', 'w').close()

		new_image_paths = m.get_new_image_paths()

		logger.info(new_image_paths)

		self.assertTrue('tmp.tmp' in new_image_paths)

		os.remove('tmp/tmp.tmp')
		os.rmdir('tmp')

	def test_update_image_paths(self):
		os.mkdir('tmp')
		open('tmp/1.tmp', 'w').close()

		m = MonitorDirectory('tmp', image_extension='.tmp')
		open('tmp/2.tmp', 'w').close()
		open('tmp/3.tmp', 'w').close()
		new_image_paths = m.get_new_image_paths()
		m.update(new_image_paths)

		self.assertEquals(m.image_paths, set(['1.tmp', '2.tmp', '3.tmp']))

		new_image_paths = m.get_new_image_paths()

		self.assertTrue(len(new_image_paths)==0)

		[os.remove(i) for i in iglob('tmp/*.tmp')]
		os.rmdir('tmp')

class PreprocessingTests(unittest.TestCase):
	def test_preprocessing(self):
		self.assertTrue(True)

	def test_raw_to_volume(self):
		with open(os.path.join(test_data_directory, 'img_rot.PixelData'), 'r') as f:
			raw_image_binary = f.read()

		raw_to_volume = RawToVolume()
		volume = raw_to_volume.run(raw_image_binary)['raw_image_volume']
		self.assertTrue(isinstance(volume, np.ndarray))

	def test_zscore(self):
		vox_zscore = VoxelZScore()
		x = np.arange(10)
		mean = np.arange(10,0,-1)
		std = np.ones(10)/2.
		z = vox_zscore.run(x, mean, std)['gm_zscore']
		self.assertTrue((z==np.array([-20.,-16.,-12.,-8.,-4.,0.,4.,8.,12.,16.])).all())

	def test_running_mean_std(self):
		running = RunningMeanStd(n=5)
		x, y = np.meshgrid(np.arange(10), np.arange(40,30,-1))
		z = x+y

		d = running.run(z[0,:])
		self.assertTrue((d['running_mean']==z[0,:]).all())
		self.assertTrue((d['running_std']==np.zeros(z.shape[1])).all())

		d = running.run(z[1,:])
		self.assertTrue((d['running_mean']==np.array([39.5,40.5,41.5,42.5,43.5,44.5,45.5,46.5,47.5,48.5])).all())

		for i in xrange(2, z.shape[0]):
			d = running.run(z[i,:])

		self.assertTrue((running.samples==z[-running.n:,:]).all())
		self.assertTrue((running.mean==np.array([ 33.,34.,35.,36.,37.,38.,39.,40.,41.,42.])).all())
		self.assertTrue((running.std==running.samples.std(0)).all())


class ImageUtilsTests(unittest.TestCase):
	def test_transform_identity(self):
		# load test image
		input_img = nbload(os.path.join(test_data_directory, 'img.nii'))
		# base image is identical
		base_path = os.path.join(test_data_directory, 'img.nii')

		output_img = transform(input_img, base_path)
		
		self.assertTrue(np.allclose(input_img.get_data(), output_img.get_data()))
		self.assertTrue(isinstance(output_img, Nifti1Image))
	
	def test_rotate(self):
		# image rotated 9 degrees in first axis
		input_img = nbload(os.path.join(test_data_directory, 'img_rot.nii'))
		base_img = nbload(os.path.join(test_data_directory, 'img.nii'))

		registered_img, xfm_pred = transform(input_img, base_img, output_transform=True)
		xfm_true = np.linalg.inv(load_afni_xfm(os.path.join(test_data_directory, 'rotmat.aff12.1D')))

		self.assertTrue(np.allclose(xfm_true, xfm_pred, rtol=0.01, atol=0.01))

	def test_motion_correct_shift(self):
		input_img = nbload(os.path.join(test_data_directory, 'img.nii'))
		# base image is identical
		base_img = nbload(os.path.join(test_data_directory, 'img.nii'))
		base_affine = base_img.affine[:]
		base_affine[0,3] += 10
		
		output_img = transform(input_img, Nifti1Image(base_img.get_data(), base_affine))
		
		self.assertTrue(np.allclose(input_img.get_data(), output_img.get_data()))


class WMDetrendTests(unittest.TestCase):
	def test_get_activity_in_mask(self):

		input_fpath = os.path.join(test_data_directory, 'img_rot.PixelData')
		with open(input_fpath, 'r') as f:
			raw_image_binary = f.read()

		ref_img = nbload(os.path.join(test_data_directory, 'img.nii'))

		wm_mask = nbload(os.path.join(test_data_directory, 'wm_mask.nii'))
		gm_mask = nbload(os.path.join(test_data_directory, 'gm_mask.nii'))

		wm = WMDetrend()
		raw_to_volume = RawToVolume()

		wm.funcref_nifti1 = ref_img
		wm.masks = { 'wm': wm_mask.get_data().astype(bool),
				         'gm': gm_mask.get_data().astype(bool)
				       }

		d = raw_to_volume.run(raw_image_binary)
		mask_activity = wm.get_activity_in_masks(d['raw_image_volume'])

		ref_wm = ref_img.get_data().T[wm.masks['wm'].T]
		self.assertTrue(np.corrcoef(mask_activity['wm'], ref_wm)[0,1]>0.99)

if __name__=='__main__':
	unittest.main()

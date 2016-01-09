import unittest
from nibabel.nifti1 import Nifti1Image
from nibabel import load as nbload, save as nbsave
from realtimefmri.data_collection import MonitorDirectory, get_example_data_directory
from realtimefmri.preprocessing import RawToVolume, WMDetrend
from realtimefmri.image_utils import transform
from realtimefmri.utils import get_subject_directory
import logging
FORMAT = '%(levelname)s: %(name)s %(funcName)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger(__name__)
import os
from glob import iglob

import time

import numpy as np

@unittest.skip('')
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

		try:
			os.mkdir('tmp')
			m = MonitorDirectory('tmp', image_extension='.tmp')

			# create dummy file
			open('tmp/tmp.tmp', 'w').close()

			new_image_paths = m.get_new_image_paths()

			logger.info(new_image_paths)

			self.assertTrue('tmp.tmp' in new_image_paths)
	
		except:
			os.remove('tmp/tmp.tmp')
			os.rmdir('tmp')

	def test_update_image_paths(self):
		try:
			os.mkdir('tmp')
			open('tmp/tmp1.tmp', 'w').close()

			m = MonitorDirectory('tmp', image_extension='.tmp')
			open('tmp/tmp2.tmp', 'w').close()
			open('tmp/tmp3.tmp', 'w').close()
			new_image_paths = m.get_new_image_paths()
			m.update(new_image_paths)

			for i in m.image_paths:
				logger.info(i)

			self.assertEquals(m.image_paths, set(['tmp1.tmp', 'tmp2.tmp', 'tmp3.tmp']))

		except:
			pass
		finally:
			[os.remove(i) for i in iglob('tmp/*.tmp')]
			os.rmdir('tmp')

class PreprocessingTests(unittest.TestCase):

	def setUp(self):
		self.test_data_directory = get_example_data_directory()
		self.test_image_fpath = os.path.join(self.test_data_directory, '3806947492785422181115102.82353.23.2.5.7011.2.21.3.1.PixelData')

		with open(self.test_image_fpath, 'r') as f:
			raw_image_binary = f.read()

		self.data_dict = { 'raw_image_binary': raw_image_binary }

	unittest.skip('not ready')
	def test_preprocessing(self):
		
		self.assertTrue(True)

	def test_raw_to_volume(self):
		raw_to_volume = RawToVolume()
		volume = raw_to_volume.run(self.data_dict)['raw_image_volume']
		self.assertTrue(isinstance(volume, np.ndarray))

class ImageUtilsTests(unittest.TestCase):
	def setUp(self):
		self.testdir = get_subject_directory('tests')

	def test_transform_identity(self):
		# load test image
		input_img = nbload(os.path.join(self.testdir, 'input_img.nii'))
		# base image is identical
		base_path = os.path.join(self.testdir, 'input_img.nii')

		output_img = transform(input_img, base_path)
		
		np.testing.assert_almost_equal(input_img.get_data(), output_img.get_data())
		self.assertTrue(isinstance(output_img, Nifti1Image))

	def test_motion_correct_shift(self):
		input_img = nbload(os.path.join(self.testdir, 'input_img.nii'))
		# base image is identical
		base_img = nbload(os.path.join(self.testdir, 'input_img.nii'))
		base_affine = base_img.affine[:]
		base_affine[0,3] += 10
		
		output_img = transform(input_img, Nifti1Image(base_img.get_data(), base_affine))
		
		np.testing.assert_almost_equal(input_img.get_data(), output_img.get_data())

@unittest.skip('not ready')
class WMDetrendTests(unittest.TestCase):
	def setUp(self):
		self.testdir = os.path.join('realtimefmri/database/tests/')

	def test_detrend(self):
		input_nifti1 = nbload(os.path.join(self.testdir, 'input_img.nii'))

		wm = WMDetrend()
		wm.detrend(input_nifti1.get_data())



if __name__=='__main__':
	unittest.main()
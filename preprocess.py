import os
import time
import argparse
from core.utils import get_log_directory
from core.preprocessing import Preprocessor

import logging
logger = logging.getLogger('preprocess')
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_log_directory(), '%s_preprocess.log'%time.strftime('%Y%m%d'))
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-20s %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Preprocess data')
	parser.add_argument('config',
		action='store',
		help='Name of configuration file')
	args = parser.parse_args()
	logger.info('Loading preprocessing pipeline from %s' % args.config)

	preproc = Preprocessor(args.config)
	preproc.run()
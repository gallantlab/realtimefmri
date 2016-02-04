#!/usr/bin/env python
import os
import time
import argparse
from core.stimulation import Stimulator
from core.utils import get_log_directory

import logging
logger = logging.getLogger('stimulate')
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_log_directory(), '%s_stimulation.log'%time.strftime('%Y%m%d'))
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
		nargs='?',
		default='stim-01',
		help='Name of configuration file')
	args = parser.parse_args()

	stim = Stimulator(args.config)
	stim.run() # this will start an infinite run loop
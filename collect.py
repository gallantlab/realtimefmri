import sys
import os
import time
import argparse
from core.collection import DataCollector
from core.utils import get_log_directory

import logging

logger = logging.getLogger('collect')
logger.setLevel(logging.DEBUG)
log_path = os.path.join(get_log_directory(), '%s_collection.log'%time.strftime('%Y%m%d'))
formatter = logging.Formatter('%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s')
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Collect data')
	parser.add_argument('-s', '--simulate', 
		action='store', 
		dest='simulate', 
		default=None, 
		help='''Simulate data collection''')
	parser.add_argument('-i', '--interval',
		action='store',
		dest='interval',
		default='2',
		help='''Interval between scans, in seconds. Only active if simulate is True''')
	parser.add_argument('-d', '--directory',
		action='store',
		dest='directory',
		default='tmp',
		help='Directory to watch')
	parser.add_argument('-p', '--parent',
		action='store_true',
		dest='parent',
		default=False,
		help='Monitor the provided directory for the first new folder, then monitor that folder for new files')
	args = parser.parse_args()
	
	try:
		interval = float(args.interval)
	except (TypeError, ValueError):
		if args.interval=='return':
			interval = args.interval
		else:
			raise ValueError

	d = DataCollector(args.directory, simulate=args.simulate, interval=interval, parent_directory=args.parent)
	d.run()

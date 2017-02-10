import os
import time
import argparse
from realtimefmri.preprocessing import Preprocessor
from realtimefmri.utils import get_logger, log_directory

log_path = os.path.join(log_directory, 'preprocess.log')
logger = get_logger('preprocess', dest=['console', log_path])

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config',
                        action='store',
                        help='Name of configuration file')
    args = parser.parse_args()
    logger.info('Loading preprocessing pipeline from %s' % args.config)

    preproc = Preprocessor(args.config)
    preproc.run()
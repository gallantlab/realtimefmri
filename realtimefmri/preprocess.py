import os
import time
import argparse
from core.preprocessing import Preprocessor
from core.utils import get_logger
logger = get_logger('preprocess', dest=['console', 'file'])

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config',
                        action='store',
                        help='Name of configuration file')
    args = parser.parse_args()
    logger.info('Loading preprocessing pipeline from %s' % args.config)

    preproc = Preprocessor(args.config)
    preproc.run()

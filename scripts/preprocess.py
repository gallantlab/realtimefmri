#!/usr/bin/env python
import os
import time
import argparse
from realtimefmri.preprocessing import Preprocessor
from realtimefmri.utils import get_logger, log_directory

if __name__=='__main__':

    logger = get_logger('preprocess', to_console=True, to_network=True)

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config',
                        action='store',
                        help='Name of configuration file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')
    args = parser.parse_args()
    logger.info('Loading preprocessing pipeline from %s' % args.config)

    preproc = Preprocessor(args.config, verbose=args.verbose)
    preproc.run()

#!/usr/bin/env python
import os
import time
import argparse
from realtimefmri.stimulating import Stimulator
from realtimefmri.utils import get_logger, log_directory


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config',
        action='store',
        nargs='?',
        default='stim-01',
        help='Name of configuration file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')
    args = parser.parse_args()

    logger = get_logger('stimulate', to_console=args.verbose, to_network=True)

    stim = Stimulator(args.config)
    stim.run() # this will start an infinite run loop

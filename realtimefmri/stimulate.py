#!/usr/bin/env python
import os
import time
import argparse
from core.stimulation import Stimulator
from core.utils import get_logger
logger = get_logger('stimulate', dest=['console', 'file'])

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

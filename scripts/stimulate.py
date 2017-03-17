#!/usr/bin/env python

import sys
import argparse
from realtimefmri.stimulating import Stimulator


def main(config, recording_id, verbose=False):

    stim = Stimulator(config, recording_id=recording_id, verbose=verbose)
    try:
        stim.run()  # this will start an infinite run loop
    except KeyboardInterrupt:
        print('shutting down stimulation')
        stim.active = False
        stim.stop()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config', action='store',
                        help='Name of configuration file')
    parser.add_argument('recording_id', action='store',
                        help='Recording name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')

    args = parser.parse_args()
    return args.config, args.recording_id, args.verbose


if __name__ == '__main__':

    main(*parse_arguments())
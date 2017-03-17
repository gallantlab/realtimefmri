#!/usr/bin/env python

import sys
import argparse
from realtimefmri.preprocessing import Preprocessor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config',
                        action='store',
                        help='Name of configuration file')
    parser.add_argument('recording_id', action='store',
                        help='Recording name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')

    args = parser.parse_args()
    return args.config, args.recording_id, args.verbose    


def main(config, recording_id, verbose=False):

    preproc = Preprocessor(config, recording_id=recording_id, verbose=verbose)
    try:
        preproc.run()
    except KeyboardInterrupt:
        print('shutting down preprocessing')
        sys.exit(0)


if __name__ == '__main__':
    main(*parse_arguments())

#!/usr/bin/env python
import argparse
from realtimefmri.stimulating import Stimulator


def main(config, recording_id, verbose=False):

    stim = Stimulator(config, recording_id=recording_id, verbose=verbose)
    stim.run()  # this will start an infinite run loop


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('config', action='store',
                        help='Name of configuration file')
    parser.add_argument('recording_id', action='store',
                        help='Recording name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')

    args = parser.parse_args()
    main(args.config, args.recording_id, args.verbose)

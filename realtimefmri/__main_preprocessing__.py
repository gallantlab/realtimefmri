#!/usr/bin/env python
"""Main entry point for preprocessing"""
import sys
import os.path as op
import time
from subprocess import Popen
import argparse
from realtimefmri.config import MODULE_DIR, get_example_data_directory
from realtimefmri.utils import get_temporary_file_name


def parse_arguments():
    """Get parser"""
    parser = argparse.ArgumentParser(description='Real-time FMRI')
    
    parser.add_argument('recording_id', action='store',
                        help='Unique recording identifier for this run')
    
    parser.add_argument('preproc_config', action='store',
                        help='Name of preprocessing configuration file')
    
    parser.add_argument('stim_config', action='store',
                        help='Name of stimulus configuration file')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose', default=False)

    args = parser.parse_args()

    return {'recording_id': args.recording_id,
            'preproc_config': args.preproc_config,
            'stim_config': args.stim_config,
            'verbose': args.verbose}


def main(recording_id=None, preproc_config=None, stim_config=None,
         verbose=None):
    """Run realtime"""

    processes = []

    # Preprocessing
    opts = [preproc_config, recording_id]
    if verbose:
        opts.append('-v')
    proc = Popen(['python', op.join(MODULE_DIR, 'preprocess.py')] +
                 opts)
    processes.append(proc)

    # Stimulation
    opts = [stim_config, recording_id]
    if verbose:
        opts.append('-v')
    proc = Popen(['python', op.join(MODULE_DIR, 'stimulate.py')] +
                 opts)
    processes.append(proc)

    try:
        input('running...')
    except KeyboardInterrupt:
        print('shutting down realtimefmri')


if __name__ == '__main__':
    main(**parse_arguments())

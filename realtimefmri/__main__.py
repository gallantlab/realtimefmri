#!/usr/bin/env python
"""Main entry point"""
import sys
import os.path as op
import time
from subprocess import Popen
import argparse
import signal
from realtimefmri.config import SCRIPTS_DIR


def get_parser():
    """Get parser"""
    parser = argparse.ArgumentParser(description='Real-time FMRI')
    parser.add_argument('recording_id', action='store',
                        help='Unique recording identifier for this run')
    parser.add_argument('dicom_dir', action='store',
                        help=("""Directory containing dicom files (or """
                              """simulation files)"""))
    parser.add_argument('preproc_config', action='store',
                        help='Name of preprocessing configuration file')
    parser.add_argument('stim_config', action='store',
                        help='Name of stimulus configuration file')
    parser.add_argument('-s', '--simulate', action='store_true',
                        default=False, dest='simulate')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')
    return parser


def run_realtimefmri(parser):
    """Run realtime"""
    args = parser.parse_args()

    try:
        processes = []

        # Logging
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'logger.py'),
                      args.recording_id])
        processes.append(proc)

        # Synchronize to TR
        opts = []
        if args.simulate:
            opts.append('-s')
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'sync.py')] +
                     opts)
        processes.append(proc)

        # Collection
        opts = ['-d', args.dicom_dir]
        if args.simulate:
            opts.append('-s')
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'collect.py')] +
                     opts)
        processes.append(proc)

        # Preprocessing
        opts = [args.preproc_config, args.recording_id]
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'preprocess.py')] +
                     opts)
        processes.append(proc)

        # Stimulation
        opts = [args.stim_config, args.recording_id]
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'stimulate.py')] +
                     opts)
        processes.append(proc)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print('killing processes')
        for proc in processes:
            proc.send_signal(signal.SIGINT)

        return 0


def main():
    """Main function"""
    return run_realtimefmri(get_parser())


if __name__ == '__main__':
    sys.exit(main())

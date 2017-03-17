#!/usr/bin/env python
"""Main entry point"""
import sys
import os.path as op
import shutil
import time
from subprocess import Popen
import argparse
import signal
from realtimefmri.config import SCRIPTS_DIR, get_example_data_directory
from realtimefmri.utils import get_temporary_file_name


def get_parser():
    """Get parser"""
    parser = argparse.ArgumentParser(description='Real-time FMRI')
    
    parser.add_argument('recording_id', action='store',
                        help='Unique recording identifier for this run')
    
    parser.add_argument('preproc_config', action='store',
                        help='Name of preprocessing configuration file')
    
    parser.add_argument('stim_config', action='store',
                        help='Name of stimulus configuration file')
    
    parser.add_argument('-d', '--dicom_directory', action='store',
                        dest='dicom_directory', default=None,
                        help=("""Directory containing dicom files (or """
                              """simulation files)"""))

    parser.add_argument('-p', '--parent_directory', action='store_true',
                        dest='parent_directory', default=False,
                        help=("""The provided directory is a parent directory.
                                 Volumes will appear in a sub-directory that
                                 will be generated after initialization of the
                                 collection script."""))
    
    parser.add_argument('-s', '--simulate_dataset', action='store',
                        dest='simulate_dataset', default=None,
                        help=("""Simulate an experiment"""))
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose', default=False)
    return parser


def run_realtimefmri(parser):
    """Run realtime"""
    args = parser.parse_args()

    try:

        if args.simulate_dataset:
            simulate_directory = get_example_data_directory(args.simulate_dataset)

        processes = []

        # Logging
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'logger.py'),
                      args.recording_id])
        processes.append(proc)

        # Synchronize to TR
        opts = []
        if args.simulate_dataset:
            opts.append('--simulate')
        proc = Popen(['python', op.join(SCRIPTS_DIR, 'sync.py')] +
                     opts)
        processes.append(proc)

        # Collection
        if args.simulate_dataset:
            temp_directory = get_temporary_file_name()
            print 'using temporary directory at {}'.format(temp_directory)
            opts = ['--directory', temp_directory]
        else:
            opts = ['--directory', args.dicom_dir]
        if args.parent_directory:
            opts.append('--parent_directory')
        if args.verbose:
            opts.append('--verbose')
        print 'starting collection'
        cmd = ['python', op.join(SCRIPTS_DIR, 'collect.py')] + opts
        print 'starting collection:\n{}'.format(' '.join(cmd))
        proc = Popen(cmd)
        processes.append(proc)

        time.sleep(1)

        # Simulate
        if args.simulate_dataset:
            opts = ['--simulate_directory', simulate_directory,
                    '--destination_directory', temp_directory]
            if args.parent_directory:
                opts.append('--parent_directory')
            if args.verbose:
                opts.append('--verbose')
            cmd = ['python', op.join(SCRIPTS_DIR, 'simulate.py')] + opts
            print 'starting simulation:\n{}'.format(' '.join(cmd))
            proc = Popen(cmd)
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
        print('****************************')
        print('* killing processes')
        print('****************************')
        for proc in processes:
            proc.send_signal(signal.SIGINT)

        return 0


def main():
    """Main function"""
    return run_realtimefmri(get_parser())


if __name__ == '__main__':
    sys.exit(main())

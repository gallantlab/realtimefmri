#!/usr/bin/env python3
import sys
import os
import os.path as op
from glob import glob
import itertools
from subprocess import Popen
import shutil
import argparse
from uuid import uuid4
import asyncio
import zmq
import zmq.asyncio

from realtimefmri import synchronize
from realtimefmri.collect import Collector
from realtimefmri.scan import Scanner
from realtimefmri.utils import get_logger
from realtimefmri.config import (SCANNER_DIR, RECORDING_DIR, PACKAGE_DIR,
                                 get_example_data_directory)


def parse_arguments():
    """Run the data collection
    """
    parser = argparse.ArgumentParser(description='Collect data')
    subcommand = parser.add_subparsers(title='subcommand', dest='subcommand')

    coll = subcommand.add_parser('collect',
                                 help="""Collect and synchronize""")
    coll.set_defaults(command_name='collect')
    coll.add_argument('recording_id', action='store', default=None,
                      help='''An identifier for this recording''')
    coll.add_argument('-d', '--directory', action='store',
                      dest='directory', default=None,
                      help=('''Directory to watch. If simulate is True,
                              simulate from this directory'''))
    coll.add_argument('-p', '--parent_directory', action='store',
                      dest='parent_directory', default=SCANNER_DIR,
                      help=('''Parent directory to watch. If provided,
                               monitor the directory for the first new folder,
                               then monitor that folder for new files'''))
    coll.add_argument('-s', '--simulate', action='store_true',
                      dest='simulate', default=False,
                      help=('''Simulate a run'''))
    coll.add_argument('-v', '--verbose', action='store_true',
                      default=False, dest='verbose',
                      help=('''Print log messages to console if true'''))

    preproc = subcommand.add_parser('preprocess',
                                    help="""Preprocess and stimulate""")
    preproc.set_defaults(command_name='preprocess')
    preproc.add_argument('recording_id', action='store',
                         help='Unique recording identifier for this run')

    preproc.add_argument('preproc_config', action='store',
                         help='Name of preprocessing configuration file')

    preproc.add_argument('stim_config', action='store',
                         help='Name of stimulus configuration file')

    preproc.add_argument('-v', '--verbose', action='store_true',
                         dest='verbose', default=False)

    simul = subcommand.add_parser('simulate',
                                  help="""Simulate a real-time experiment""")
    simul.set_defaults(command_name='simulate')
    simul.add_argument('simulate_dataset', action='store')

    args = parser.parse_args()

    return args


def collect(recording_id, directory=None, parent_directory=None, simulate=False,
            verbose=False):
    """Collect volumes and synchronize with the scanner
    """

    log_path = op.join(RECORDING_DIR, recording_id, 'recording.log')
    if not op.exists(op.dirname(log_path)):
        os.makedirs(op.dirname(log_path))
        print('making recording directory {}'.format(op.dirname(log_path)))
    print('saving log file to {}'.format(log_path))

    logger = get_logger('root', to_file=log_path, to_console=verbose)

    loop = asyncio.get_event_loop()

    tasks = []

    logger.info('starting synchronize.Synchronizer')
    sync = synchronize.Synchronizer(verbose=True, loop=loop)
    tasks.append(sync.run())

    logger.info('starting scanner')
    scan = Scanner(simulate=simulate, verbose=True, loop=loop)
    tasks.append(scan.run())

    logger.info('starting collector')
    coll = Collector(directory=directory, parent_directory=parent_directory,
                     verbose=True, loop=loop)
    tasks.append(coll.run())

    try:
        logger.info('starting tasks')
        loop.run_until_complete(asyncio.gather(*tasks))
    except KeyboardInterrupt:
        logger.info('shutting down')
        [task.cancel() for task in tasks]
        loop.close()


def preprocess(recording_id, preproc_config=None, stim_config=None,
               verbose=None):
    """Run realtime
    """

    processes = []

    # Preprocessing
    opts = [preproc_config, recording_id]
    if verbose:
        opts.append('-v')
    proc = Popen(['python3', op.join(PACKAGE_DIR, 'preprocess.py')] +
                 opts)
    processes.append(proc)

    # Stimulation
    opts = [stim_config, recording_id]
    if verbose:
        opts.append('-v')
    proc = Popen(['python3', op.join(PACKAGE_DIR, 'stimulate.py')] +
                 opts)
    processes.append(proc)

    try:
        input('')
    except KeyboardInterrupt:
        print('shutting down realtimefmri')


def simulate(simulate_dataset):
    """Simulate sync pulses and image acquisition
    """

    ex_directory = get_example_data_directory(simulate_dataset)
    paths = glob(op.join(ex_directory, '*.dcm'))

    dest_directory = op.join(SCANNER_DIR, str(uuid4()))
    os.makedirs(dest_directory)
    print('Simulated {} volumes appearing in {}'.format(len(paths),
                                                        dest_directory))

    try:
        for path in itertools.cycle(paths):
            input('>>> press 5 for TTL, then enter for new image')
            new_path = op.join(dest_directory, str(uuid4()) + '.dcm')
            shutil.copy(path, new_path)
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(dest_directory)


def main():
    args = parse_arguments()
    if args.subcommand == 'collect':
        print(args)
        collect(args.recording_id, directory=args.directory,
                parent_directory=args.parent_directory, simulate=args.simulate,
                verbose=args.verbose)
    elif args.subcommand == 'preprocess':
        preprocess(args.recording_id, preproc_config=args.preproc_config,
                   stim_config=args.stim_config, verbose=args.verbose)
        print(sys.path)
    elif args.subcommand == 'simulate':
        simulate(args.simulate_dataset)


if __name__ == '__main__':
    main()

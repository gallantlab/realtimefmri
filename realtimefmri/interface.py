#!/usr/bin/env python3
import os
import os.path as op

from glob import glob
import itertools
from subprocess import Popen
import shutil
from uuid import uuid4
import asyncio

from realtimefmri import synchronize
from realtimefmri.collect import Collector
from realtimefmri.scan import Scanner
from realtimefmri.utils import get_logger
from realtimefmri.config import (SCANNER_DIR, RECORDING_DIR, PACKAGE_DIR,
                                 get_dataset)


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

    ex_directory = get_dataset(simulate_dataset)
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

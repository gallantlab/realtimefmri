from os import makedirs
import os.path as op
from subprocess import Popen
import argparse
import asyncio
import zmq
import zmq.asyncio

from realtimefmri.synchronize import Synchronizer
from realtimefmri.collect import Collector
from realtimefmri.scan import Scanner
from realtimefmri.utils import get_logger
from realtimefmri.config import RECORDING_DIR, MODULE_DIR


def parse_arguments():
    """Run the data collection
    """
    parser = argparse.ArgumentParser(description='Collect data')
    subcommand = parser.add_subparsers(title='subcommand', dest='subcommand')
    console = subcommand.add_parser('console',
                                    help="""Collect and synchronize""")

    console.set_defaults(command_name='console')
    console.add_argument('recording_id', action='store', default=None,
                         help='''An identifier for this recording''')
    console.add_argument('-d', '--directory', action='store',
                         dest='directory', default=None,
                         help=('''Directory to watch. If simulate is True,
                                  simulate from this directory'''))
    console.add_argument('-p', '--parent_directory', action='store',
                         dest='parent_directory', default=None,
                         help=('''Parent directory to watch. If provided,
                                  monitor the directory for the first new folder,
                                  then monitor that folder for new files'''))
    console.add_argument('-s', '--simulate', action='store_true',
                         dest='simulate', default=False,
                         help=('''Simulate a run'''))
    console.add_argument('-v', '--verbose', action='store_true',
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

    args = parser.parse_args()
    
    return args

def console(recording_id, directory=None, parent_directory=None, simulate=False,
            verbose=False):

    log_path = op.join(RECORDING_DIR, recording_id, 'recording.log')
    if not op.exists(op.dirname(log_path)):
        makedirs(op.dirname(log_path))
        print('making recording directory {}'.format(op.dirname(log_path)))
    print('saving log file to {}'.format(log_path))

    logger = get_logger('root', to_file=log_path, to_console=verbose)

    loop = zmq.asyncio.ZMQEventLoop()
    
    tasks = []
    
    logger.info('starting synchronizer')
    sync = Synchronizer(verbose=True, loop=loop)
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


def main(args):

    if args.subcommand == 'console':
        console(args.recording_id, directory=args.directory,
                parent_directory=args.parent_directory, simulate=args.simulate,
                verbose=args.verbose)
    elif args.subcommand == 'preprocess':
        preprocess(args.recording_id, preproc_config=args.preproc_config,
                   stim_config=args.stim_config, verbose=args.verbose)


if __name__ == '__main__':
    main(parse_arguments())
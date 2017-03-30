from os import makedirs
import os.path as op
import argparse
import asyncio
import zmq
import zmq.asyncio

from realtimefmri.synchronize import Synchronizer
from realtimefmri.collect import Collector
from realtimefmri.scan import Scanner
from realtimefmri.utils import get_logger
from realtimefmri.config import RECORDING_DIR

def parse_arguments():
    """Run the data collection
    """
    parser = argparse.ArgumentParser(description='Collect data')
    parser.add_argument('recording_id', action='store', default=None,
                       help='''An identifier for this recording''')
    parser.add_argument('-d', '--directory', action='store',
                        dest='directory', default=None,
                        help=('''Directory to watch. If simulate is True,
                                 simulate from this directory'''))
    parser.add_argument('-p', '--parent_directory', action='store',
                        dest='parent_directory', default=None,
                        help=('''Parent directory to watch. If provided,
                                 monitor the directory for the first new folder,
                                 then monitor that folder for new files'''))
    parser.add_argument('-s', '--simulate', action='store_true',
                        dest='simulate', default=False,
                        help=('''Simulate a run'''))
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose',
                        help=('''Print log messages to console if true'''))

    args = parser.parse_args()
    
    return {'recording_id': args.recording_id,
            'directory': args.directory,
            'parent_directory': args.parent_directory,
            'simulate': args.simulate,
            'verbose': args.verbose}

def main(recording_id=None, directory=None, parent_directory=None,
         simulate=False, verbose=False):

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


if __name__ == '__main__':
    main(**parse_arguments())
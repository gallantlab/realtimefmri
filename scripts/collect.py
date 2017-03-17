#!/usr/bin/env python

'''Script to run data collection
'''

import argparse
from realtimefmri.collecting import Collector
from realtimefmri.utils import get_logger
from realtimefmri.config import get_example_data_directory


def main():
    '''Run the data collection
    '''
    parser = argparse.ArgumentParser(description='Collect data')
    parser.add_argument('-d', '--directory',
                        action='store',
                        dest='directory',
                        default='tmp',
                        help=('''Directory to watch. If simulate is True,
                                 simulate from this directory'''))
    parser.add_argument('-p', '--parent_directory',
                        action='store_true',
                        dest='parent_directory',
                        default=False,
                        help=('''Default False. Use with `-d`. If true, monitor
                                 the provided directory for the first new
                                 folder, then monitor that folder for new
                                 files'''))
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose',
                        help=('''Print log messages to console if true'''))
    args = parser.parse_args()

    logger = get_logger('collect', to_console=args.verbose, to_network=True)

    data_collector = Collector(directory=args.directory,
                               parent_directory=args.parent_directory,
                               verbose=args.verbose)

    logger.info('running collection')
    data_collector.run()


if __name__ == "__main__":
    main()

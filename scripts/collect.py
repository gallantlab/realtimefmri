#!/usr/bin/env python
import sys
import os
import time
import argparse
from realtimefmri.collecting import DataCollector
from realtimefmri.utils import get_logger, log_directory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Collect data')
    parser.add_argument('-s', '--simulate', 
        action='store_true', 
        dest='simulate',
        default=None,
        help='''default False. Simulate from directory''')
    parser.add_argument('-i', '--interval',
        action='store',
        dest='interval',
        default='2',
        help=('''Default 2. Designate interval between scans. 
                `return`, simulate image acquisition every time the return key is pressed, 
                int, interval in seconds between image acquisition. 
                Only active if simulate is True'''))
    parser.add_argument('-d', '--directory',
        action='store',
        dest='directory',
        default='tmp',
        help='Directory to watch. If simulate is True, simulate from this directory')
    parser.add_argument('-p', '--parent',
        action='store_true',
        dest='parent',
        default=False,
        help=('''Default False. Use with `-d`. If true, monitor the provided
                 directory for the first new folder, then monitor that folder
                 for new files'''))
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose',
                        help=('''Print log messages to console if true'''))
    args = parser.parse_args()


    logger = get_logger('collect', to_console=args.verbose, to_network=True)
    
    try:
        interval = float(args.interval)
    except (TypeError, ValueError):
        if args.interval in ['return', 'sync']:
            interval = args.interval
        else:
            raise ValueError, '''Interval must be an integer (in seconds), "return", or "sync"'''

    d = DataCollector(directory=args.directory, simulate=args.simulate,
                      interval=interval, parent_directory=args.parent, verbose=args.verbose)
    d.run()

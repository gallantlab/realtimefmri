import sys
import os
import time
import argparse
from core.collection import DataCollector
from core.utils import get_logger
logger = get_logger('collect', dest=['console', 'file'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Collect data')
    parser.add_argument('-s', '--simulate', 
        action='store', 
        dest='simulate', 
        default=None,
        help='''Simulate data collection''')
    parser.add_argument('-i', '--interval',
        action='store',
        dest='interval',
        default='2',
        help='''Designate interval between scans. \n
        `return`, simulate image acquisition every time the return key is pressed, \n
        `sync`, synchronize to TTL pulse, \n
        int, interval in seconds between image acquisition. \n
        Only active if simulate is True''')
    parser.add_argument('-d', '--directory',
        action='store',
        dest='directory',
        default='tmp',
        help='Directory to watch')
    parser.add_argument('-p', '--parent',
        action='store_true',
        dest='parent',
        default=None,
        help='Monitor the provided directory for the first new folder, then monitor that folder for new files')
    args = parser.parse_args()
    
    try:
        interval = float(args.interval)
    except (TypeError, ValueError):
        if args.interval in ['return', 'sync']:
            interval = args.interval
        else:
            raise ValueError, '''Interval must be an integer (in seconds), "return", or "sync"'''

    d = DataCollector(directory=args.directory, simulate=args.simulate,
                      interval=interval, parent_directory=args.parent)
    d.run()

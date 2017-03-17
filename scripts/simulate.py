#!/usr/bin/env python

'''Script to run data collection
'''

import sys
import argparse
from realtimefmri.collecting import Simulator
from realtimefmri.utils import get_logger


def main():
    '''Run the data collection
    '''
    parser = argparse.ArgumentParser(description='Simulate data collection')
    parser.add_argument('-i', '--interval', action='store',
                        dest='interval', default=2,
                        help=('''Default 2. Designate interval between scans.
                                `return`, simulate image acquisition every time
                                the return key is pressed, int, interval in
                                seconds between image acquisition. Only active
                                if simulate is True'''))
    parser.add_argument('-s', '--simulate_directory', action='store',
                        dest='simulate_directory', default=None,
                        help=('''Directory to simulate from'''))
    parser.add_argument('-d', '--destination_directory', action='store',
                        dest='destination_directory', default=None)
    parser.add_argument('-p', '--parent_directory', action='store_true',
                        dest='parent_directory', default=False)
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose', default=False,
                        help=('''Print log messages to console if true'''))
    args = parser.parse_args()

    get_logger('simulate', to_console=args.verbose, to_network=True)

    simulator = Simulator(simulate_directory=args.simulate_directory,
                          destination_directory=args.destination_directory,
                          parent_directory=args.parent_directory,
                          interval=args.interval,
                          verbose=args.verbose)
    try:
        simulator.run()
    except KeyboardInterrupt:
        print('shutting down simulation')
        simulator.active = False
        simulator.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()

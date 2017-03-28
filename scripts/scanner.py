#!/usr/bin/env python

import sys
import argparse
from realtimefmri.scanning import Scanner


def main(simulate=False, verbose=False):
    log_dest = ['network']
    if verbose:
        log_dest.append('console')

    scanner = Scanner(simulate=simulate, log_dest=log_dest)
    
    try:
        scanner.run()
    except KeyboardInterrupt:
        print('shutting down syncing')
        sys.exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Record TR times')
    parser.add_argument('-s', '--simulate', action='store_true',
                        dest='simulate', default=False)
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose', default=False)

    args = parser.parse_args()

    return args.simulate, args.verbose


if __name__ == '__main__':
    main(*parse_arguments())

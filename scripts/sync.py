#!/usr/bin/env python
import argparse
from realtimefmri.scanner import Scanner

def main(simulate=False, verbose=False):
    if args.verbose:
        log_dest.append('console')

    scanner = Scanner(simulate=simulate, log_dest=['network'])
    scanner.run()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Record TR times')
    parser.add_argument('-s', '--simulate', action='store_true',
                        dest='simulate', default=False)
    parser.add_argument('-v', '--verbose', action='store_true',
                        dest='verbose', default=False)

    args = parser.parse_args()    


    main(simulate=args.simulate, verbose=args.verbose)

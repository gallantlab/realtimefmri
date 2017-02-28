#!/usr/bin/env python
import sys
import time
from subprocess import Popen
import argparse
import signal

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('experiment_id', action='store')
    parser.add_argument('dicom_dir', action='store',
                        help='Directory containing dicom files (or simulation files)')
    parser.add_argument('preproc_config', action='store',
                        help='Name of preprocessing configuration file')
    parser.add_argument('stim_config', action='store',
                        help='Name of stimulus configuration file')
    parser.add_argument('-s', '--simulate', action='store_true',
                        default=False, dest='simulate')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, dest='verbose')
    args = parser.parse_args()


    try:
        proc = []
        p = Popen(['python', 'logger.py', args.experiment_id])
        proc.append(p)

        ## Synchronize to TR
        opts = []
        if args.simulate:
            opts.append('-s')
        p = Popen(['python', 'sync.py'] + opts)
        proc.append(p)

        ## Collection
        opts = ['-d', args.dicom_dir]
        if args.simulate:
            opts.append('-s')
        p = Popen(['python', 'collect.py'] + opts)
        proc.append(p)

        ## Preprocessing
        p = Popen(['python', 'preprocess.py', args.preproc_config])
        proc.append(p)

        ## Stimulation
        p = Popen(['python', 'stimulate.py', args.stim_config])
        proc.append(p)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print 'killing processes'
        [p.send_signal(signal.SIGINT) for p in proc]
        sys.exit(0)

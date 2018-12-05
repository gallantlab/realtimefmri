#!/usr/bin/env python3
import argparse
from realtimefmri import collect_ttl, collect_volumes, collect
from realtimefmri import preprocess
from realtimefmri import web_interface
from realtimefmri import config


def parse_arguments():
    """Run the data collection
    """
    parser = argparse.ArgumentParser(description='Collect data')
    subcommand = parser.add_subparsers(title='subcommand', dest='subcommand')

    ttl = subcommand.add_parser('collect_ttl', help="""Synchronize to TTL pulses""")
    ttl.set_defaults(command_name='collect_ttl')
    ttl.add_argument('source', action='store', default='keyboard',
                     help='''TTL source. keyboard, serial, redis, or simulate''')

    vol = subcommand.add_parser('collect_volumes', help="""Monitor for arrival of volumes""")
    vol.set_defaults(command_name='collect_volumes')

    coll = subcommand.add_parser('collect',
                                 help="""Collect and synchronize""")
    coll.set_defaults(command_name='collect')
    coll.add_argument('-v', '--verbose', action='store_true',
                      default=True, dest='verbose',
                      help=('''Print log messages to console if true'''))

    preproc = subcommand.add_parser('preprocess',
                                    help="""Preprocess and stimulate""")
    preproc.set_defaults(command_name='preprocess')
    preproc.add_argument('recording_id', action='store',
                         help='Unique recording identifier for this run')
    preproc.add_argument('preproc_config', action='store',
                         help='Name of preprocessing configuration file')
    preproc.add_argument('-v', '--verbose', action='store_true',
                         dest='verbose', default=True)

    simul = subcommand.add_parser('simulate',
                                  help="""Simulate a real-time experiment""")
    simul.set_defaults(command_name='simulate')
    simul.add_argument('simulate_dataset', action='store')

    control = subcommand.add_parser('web_interface',
                                    help="""Launch web interface for controlling real-time
                                            experiments""")
    control.set_defaults(command_name='web_interface')

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if args.subcommand == 'collect_ttl':
        collect_ttl.collect_ttl(source=args.source)

    elif args.subcommand == 'collect_volumes':
        collect_volumes.collect_volumes()

    elif args.subcommand == 'collect':
        collect.collect(args.verbose)

    elif args.subcommand == 'preprocess':
        preprocess.preprocess(args.recording_id, args.preproc_config, verbose=args.verbose)

    elif args.subcommand == 'web_interface':
        print(web_interface)
        print(dir(web_interface))
        web_interface.index.serve()


if __name__ == '__main__':
    main()

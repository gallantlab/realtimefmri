#!/usr/bin/env python3
import argparse
from realtimefmri import control_panel
from realtimefmri import dashboard
from realtimefmri.interface import collect, preprocess, simulate
from realtimefmri.config import SCANNER_DIR


def parse_arguments():
    """Run the data collection
    """
    parser = argparse.ArgumentParser(description='Collect data')
    subcommand = parser.add_subparsers(title='subcommand', dest='subcommand')

    coll = subcommand.add_parser('collect',
                                 help="""Collect and synchronize""")
    coll.set_defaults(command_name='collect')
    coll.add_argument('recording_id', action='store', default=None,
                      help='''An identifier for this recording''')
    coll.add_argument('-d', '--directory', action='store',
                      dest='directory', default=None,
                      help=('''Directory to watch. If simulate is True,
                              simulate from this directory'''))
    coll.add_argument('-p', '--parent_directory', action='store',
                      dest='parent_directory', default=SCANNER_DIR,
                      help=('''Parent directory to watch. If provided,
                               monitor the directory for the first new folder,
                               then monitor that folder for new files'''))
    coll.add_argument('-s', '--simulate', action='store_true',
                      dest='simulate', default=False,
                      help=('''Simulate a run'''))
    coll.add_argument('-v', '--verbose', action='store_true',
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

    simul = subcommand.add_parser('simulate',
                                  help="""Simulate a real-time experiment""")
    simul.set_defaults(command_name='simulate')
    simul.add_argument('simulate_dataset', action='store')

    control = subcommand.add_parser('control_panel',
                                    help="""Launch web interface for controlling real-time
                                            experiments""")

    dashb = subcommand.add_parser('dashboard',
                                  help="""Launch web interface for viewing results from real-time 
                                          experiments""")
    dashb.add_argument('--host', default='localhost', dest='host', action='store')
    dashb.add_argument('--port', default=8050, dest='port', action='store')
    dashb.add_argument('--redis_host', default='localhost', dest='redis_host', action='store')
    dashb.add_argument('--redis_port', default=6379, dest='redis_port', action='store')

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if args.subcommand == 'collect':
        print(args)
        collect(args.recording_id, directory=args.directory,
                parent_directory=args.parent_directory, simulate=args.simulate,
                verbose=args.verbose)

    elif args.subcommand == 'preprocess':
        preprocess(args.recording_id, preproc_config=args.preproc_config,
                   stim_config=args.stim_config, verbose=args.verbose)

    elif args.subcommand == 'simulate':
        simulate(args.simulate_dataset)

    elif args.subcommand == 'control_panel':
        control_panel.main()

    elif args.subcommand == 'dashboard':
        dashboard.main(host=args.host, port=args.port,
                       redis_host=args.redis_host, redis_port=args.redis_port)


if __name__ == '__main__':
    main()

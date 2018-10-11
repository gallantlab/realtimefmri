#!/usr/bin/env python3
import argparse
from realtimefmri import control_panel
from realtimefmri import dashboard
from realtimefmri import interfaces
from realtimefmri import synchronize
from realtimefmri.config import SCANNER_DIR


def parse_arguments():
    """Run the data collection
    """
    parser = argparse.ArgumentParser(description='Collect data')
    subcommand = parser.add_subparsers(title='subcommand', dest='subcommand')

    sync = subcommand.add_parser('synchronize', help="""Synchronize to TTL pulses""")
    sync.set_defaults(command_name='synchronize')
    sync.add_argument('source', action='store', default='keyboard',
                      help='''TTL source. keyboard, serial, redis, zmq, or simulate''')

    coll = subcommand.add_parser('collect',
                                 help="""Collect and synchronize""")
    coll.set_defaults(command_name='collect')
    coll.add_argument('recording_id', action='store', default=None,
                      help='''An identifier for this recording''')
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
    control.set_defaults(command_name='control_panel')

    dashb = subcommand.add_parser('dashboard',
                                  help="""Launch web interface for viewing results from real-time 
                                          experiments""")
    dashb.set_defaults(command_name='dashboard')
    dashb.add_argument('--host', default='0.0.0.0', dest='host', action='store')
    dashb.add_argument('--port', default=8050, dest='port', action='store')
    dashb.add_argument('--redis_host', default='redis', dest='redis_host', action='store')
    dashb.add_argument('--redis_port', default=6379, dest='redis_port', action='store')

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if args.subcommand == 'synchronize':
        synchronize.Synchronize(ttl_source=args.source).run()

    elif args.subcommand == 'collect':
        interfaces.collect(args.recording_id, verbose=args.verbose)

    elif args.subcommand == 'preprocess':
        interfaces.preprocess(args.recording_id, preproc_config=args.preproc_config,
                              stim_config=args.stim_config, verbose=args.verbose)

    elif args.subcommand == 'control_panel':
        control_panel.main()

    elif args.subcommand == 'dashboard':
        dashboard.main(host=args.host, port=args.port,
                       redis_host=args.redis_host, redis_port=args.redis_port)


if __name__ == '__main__':
    main()

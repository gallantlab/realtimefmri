import sys
import time
import zmq
import serial
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Receive sync pulses from scanner.')
    parser.add_argument('-s', '--simulate',
        action='store_true',
        dest='simulate',
        default=False,
        help='Simulate image acquisition')
    parser.add_argument('-p', '--port',
        action='store',
        dest='port',
        default=5554,
        help='''Port to which zmq sync messages are pushed''')

    args = parser.parse_args()

    img_msg = '' # what is this
    ctx = zmq.Context()
    s = ctx.socket(zmq.PUSH)
    s.bind('tcp://*:%d' % args.port)

    try:
        if args.simulate:
            raw_input('press return to start simulated acquisition >>')
            print 'starting'
            while True:
                s.send('time acquiring image')
                time.sleep(2)
        else:
            ser = serial.Serial('/dev/ttyUSB0')
            while True:
                msg = ser.read()
                print msg
                if msg==img_msg:
                    s.send('time acquiring image')
    except KeyboardInterrupt:
            print 'stopped'
            sys.exit(0)

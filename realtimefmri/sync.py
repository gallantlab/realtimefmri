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

	args = parser.parse_args()

	img_msg = '' # what is this
	ctx = zmq.Context()
	s = ctx.socket(zmq.PUB)
	s.bind('tcp://*:5554')

	if args.simulate:
		raw_input('press return to start simulated acquisition >>')
		try:
			print 'starting'
			while True:
				s.send('time acquiring image')
				time.sleep(2)
		except KeyboardInterrupt:
			print 'stopped'
			sys.exit(0)
	else:
		ser = serial.Serial('/dev/ttyUSB0')
		while True:
			msg = ser.read()
			print msg
			if msg==img_msg:
				s.send('time acquiring image')
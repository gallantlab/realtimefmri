import time
import struct
import argparse
import zmq

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Serve the clock')
	parser.add_argument('-p', '--port', action='store',
						dest='port', default=5550)

	args = parser.parse_args()

	ctx = zmq.Context()
	s = ctx.socket(zmq.REP)
	s.bind('tcp://*:{}'.format(args.port))
	print 'Serving time on port {}'.format(args.port)
	while True:
		s.recv()
		s.send(struct.pack('d', time.time()))
		time.sleep(0.01)
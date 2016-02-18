import time
import zmq

ctx = zmq.Context()
s = ctx.socket(zmq.PUSH)
s.bind('tcp://localhost:5554')
while True:
	s.send('acquiring image')
	time.sleep(2)
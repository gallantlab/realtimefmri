import zmq
import numpy as np
from realtimefmri.utils import parse_message
from realtimefmri.config import STIM_ADDRESS


context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://localhost:%d' % STIM_ADDRESS)
socket.setsockopt(zmq.SUBSCRIBE, b'')

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

while True:

    active = dict(poller.poll())
    if socket in active and active[socket] == zmq.POLLIN:
        message = socket.recv_multipart()
        topic, sync_time, data = parse_message(message)
        print(len(np.fromstring(data, 'float32')))

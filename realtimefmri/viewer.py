import pickle
import time
import warnings

import numpy as np
import redis

import cortex
from realtimefmri import config
from realtimefmri.utils import get_logger

logger = get_logger('realtimefmri.viewer', to_console=True, to_network=False,
                    to_file=True)

r = redis.StrictRedis(config.REDIS_HOST)


class PyCortexViewer(object):
    bufferlen = 15

    def __init__(self, surface, transform, mask_type='thick', vmin=-2., vmax=2.):
        if mask_type == '':
            data = np.zeros((self.bufferlen, 30, 100, 100), 'float32')
        else:
            npts = cortex.db.get_mask(surface, transform, mask_type).sum()
            data = np.zeros((self.bufferlen, npts), 'float32')

        vol = cortex.Volume(data, surface, transform, vmin=vmin, vmax=vmax)
        logger.debug('Starting pycortex viewer')
        server = cortex.webgl.show(vol, open_browser=False, autoclose=False, port=8051)
        logger.debug('Started pycortex viewer %s %s %s', surface, transform, mask_type)
        view = server.get_client()
        logger.debug('Client connected')

        self.surface = surface
        self.transform = transform
        self.mask_type = mask_type

        self.view = view
        self.active = True
        self.i = 0

    def update_viewer(self, volume):
        logger.info(f"Updating pycortex viewer")
        volume = volume.astype('float32')
        volume = cortex.Volume(volume, self.surface, self.transform)
        mosaic, _ = cortex.mosaic(volume.volume[0], show=False)
        i, = self.view.setFrame()
        logger.debug("""i, = self.view.setFrame() %f""", i)

        if isinstance(i, (int, float)):
            i = round(i)
            logger.debug("i = round(i) %f", i)
            new_frame = (i + 1) % self.bufferlen
            logger.debug("""new_frame %f""", new_frame)

            self.view.dataviews.data.data[0]._setData(new_frame, mosaic)
            logger.debug("""self.view.dataviews.data.data[0]._setData(new_frame, mosaic)""")

            logger.debug("Play %f", i)
            self.view.playpause('play')
            time.sleep(1)
            self.view.playpause('pause')
            self.view.setFrame(i + 0.99)
            logger.debug("Pause %f", i + 0.99)
        else:
            warnings.warn(f'setFrame returned {i}')

    def run(self):
        subscriber = r.pubsub()
        subscriber.subscribe('viewer')
        logger.info('Listening for volumes')
        for message in subscriber.listen():
            if message['type'] == 'message':
                vol = pickle.loads(message['data'])
                self.update_viewer(vol)


def serve(surface, transform, mask_type, vmin=-2, vmax=2):
    viewer = PyCortexViewer(surface, transform, mask_type, vmin, vmax)
    viewer.run()

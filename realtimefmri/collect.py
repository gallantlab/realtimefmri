import os.path as op
import pickle
import struct

import redis

from realtimefmri import config, image_utils
from realtimefmri.utils import get_logger


r = redis.StrictRedis(config.REDIS_HOST)


def collect(verbose=True):
    """Continuously monitor for incoming volumes, merge with TTL timestamps, and send to
    preprocessor
    """
    logger = get_logger('collector', to_console=verbose, to_network=True)
    logger.info('data collector initialized')

    redis_client = redis.StrictRedis(config.REDIS_HOST)
    volume_subscriber = redis_client.pubsub()
    volume_subscriber.subscribe('volume')

    image_number = 0
    for message in volume_subscriber.listen():
        if message['type'] == 'message':

            new_volume_path = message['data'].decode('utf-8')
            new_volume_path = op.join(config.SCANNER_DIR, new_volume_path)
            logger.info('New volume %s', new_volume_path)
            timestamp = redis_client.rpop('timestamp')
            timestamp = struct.unpack('d', timestamp)[0]
            logger.info('Collected at %d', timestamp)

            nii = image_utils.dicom_to_nifti(new_volume_path)
            timestamped_volume = {'image_number': image_number, 'time': timestamp, 'volume': nii}

            logger.debug('%s %s', op.basename(new_volume_path), str(nii.shape))
            redis_client.publish('timestamped_volume', pickle.dumps(timestamped_volume))

            r.set('image_number', pickle.dumps(image_number))
            image_number += 1

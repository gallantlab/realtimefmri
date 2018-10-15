#!/usr/bin/env python3
import os.path as op
import struct
import pickle
import redis
import pydicom
from dicom2nifti import convert_siemens
from realtimefmri.utils import get_logger
from realtimefmri import config


def collect(verbose=True):
    """Continuously monitor for incoming volumes, merge with TTL timestamps, and send to 
    preprocessor
    """
    logger = get_logger('collector', to_console=verbose, to_network=True)
    logger.info('data collector initialized')

    redis_client = redis.StrictRedis(config.REDIS_HOST)
    volume_subscriber = redis_client.pubsub()
    volume_subscriber.subscribe('volume')

    for image_number, message in enumerate(volume_subscriber.listen()):
        if isinstance(message['data'], (str, bytes)):
            new_volume_path = message['data'].decode('utf8')
            logger.info('New volume {}'.format(new_volume_path))
            timestamp = redis_client.rpop('timestamp')
            timestamp = struct.unpack('d', timestamp)[0]
            logger.info('Collected at {}'.format(timestamp))

            dcm = [pydicom.read_file(new_volume_path)]
            nii = convert_siemens.dicom_to_nifti(dcm, None)['NII']

            logger.debug('%s %s', op.basename(new_volume_path), str(nii.shape))
            redis_client.publish('timestamped_volume', pickle.dumps([image_number, timestamp, nii]))

"""Data collection code
"""
import itertools
import os
import os.path as op
import shutil
from uuid import uuid4

import pyinotify
import redis

from realtimefmri import config
from realtimefmri.utils import get_logger

logger = get_logger('collect_volumes', to_console=True, to_network=True)


def collect_volumes():
    watch_manager = pyinotify.WatchManager()  # Watch Manager
    handler = DirectoryMonitor()
    notifier = pyinotify.Notifier(watch_manager, handler)
    mask = pyinotify.IN_CLOSE_WRITE
    logger.info('Watching {}'.format(config.SCANNER_DIR))
    watch_manager.add_watch(config.SCANNER_DIR, mask, auto_add=True)

    notifier.loop()


class DirectoryMonitor(pyinotify.ProcessEvent):
    def __init__(self, extension='.dcm'):
        self.redis_client = redis.StrictRedis(config.REDIS_HOST)
        self.extension = extension

    def process_IN_CLOSE_WRITE(self, event):
        logger.info('Volume {}'.format(event.pathname))
        if (not op.isdir(event.pathname)) and (op.splitext(event.pathname)[1] == self.extension):
            self.redis_client.publish('volume', event.pathname)


def simulate_volumes(dataset):
    dest_directory = op.join(config.SCANNER_DIR, str(uuid4()))
    os.makedirs(dest_directory)
    paths = config.get_dataset_volume_paths(dataset)
    for i, path in enumerate(itertools.cycle(paths)):
        dest_path = op.join(dest_directory, f"IM{i:04}.dcm")
        shutil.copy(path, dest_path)
        yield path

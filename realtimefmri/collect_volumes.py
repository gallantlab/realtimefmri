import os
import os.path as op
import shutil
import itertools
from uuid import uuid4
import pyinotify
import redis
from realtimefmri import config


class VolumeCollector(pyinotify.ProcessEvent):
    def __init__(self, extension='.dcm', verbose=True):
        self.redis_client = redis.Redis(config.REDIS_HOST)
        self.extension = extension
        self.verbose = verbose

    def process_IN_CLOSE_WRITE(self, event):
        if (not op.isdir(event.pathname)) and (op.splitext(event.pathname)[1] == self.extension):
            self.redis_client.publish('volume', event.pathname)
            if self.verbose:
                print("Detected:", event.pathname)


def simulate_volumes(dataset):
    dest_directory = op.join(config.SCANNER_DIR(str(uuid4())))
    os.makedirs(dest_directory)
    paths = config.get_dataset_volume_paths(dataset)
    for path in itertools.cycle(paths):
        dest_path = op.join(dest_directory, str(uuid4() + '.dcm'))
        shutil.copy(path, dest_path)
        yield path


def collect_volumes(verbose=True):
    watch_manager = pyinotify.WatchManager()  # Watch Manager
    handler = VolumeCollector(verbose=verbose)
    notifier = pyinotify.Notifier(watch_manager, handler)
    mask = pyinotify.IN_CLOSE_WRITE
    print('Watching {}'.format(config.SCANNER_DIR))
    watch_manager.add_watch(config.SCANNER_DIR, mask, auto_add=True)

    notifier.loop()

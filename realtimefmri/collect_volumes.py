"""Data collection code
"""
import os
import os.path as op
import shutil
import time
import itertools
from uuid import uuid4
import pyinotify
import redis
from realtimefmri import config
from realtimefmri.utils import get_logger


logger = get_logger('collect_volumes', to_console=True, to_network=True)


def collect_volumes_poll(directory=None, parent_directory=None, extension=None):
    """Continuously read the monitor directory and add publish new files as they 
    are detected.
    """
    if (directory is None) and (parent_directory is None):
        raise ValueError("Must provide either directory or parent_directory.")

    redis_client = redis.StrictRedis(config.REDIS_HOST)

    if parent_directory:
        monitor = DirectoryMonitorPoll(parent_directory, extension='/')
        logger.debug(f'Detecting next subfolder in {parent_directory}')
        directory = op.join(parent_directory, next(monitor.yield_new_paths()))

    monitor = DirectoryMonitorPoll(directory, extension=extension)

    for new_path in monitor.yield_new_paths():
        logger.info('volume %s', new_path)
        redis_client.publish('volume', new_path)

    time.sleep(0.1)


class DirectoryMonitorPoll(object):
    """
    Monitor the file contents of a directory

    Example
    -------
    m = DirectoryMonitorPoll(dir_path)
    # add a file to that directory
    new_paths = m.get_new_contents()
    # use the new images
    # update image paths list to contain newly acquired images
    m.update(new_paths)
    # no images added
    new_paths = m.get_new_contents()
    len(new_paths)==0 # True
    """
    def __init__(self, directory, pattern=None, extension='.dcm'):
        if extension == '/':
            self._is_valid = self._is_valid_directories
        else:
            self._is_valid = self._is_valid_files

        self.directory = directory
        self.extension = extension
        self.directory_contents = set()
        self.directory_contents = self.get_new_contents()
        self.last_modtime = 0

    def _is_valid_directories(self, val):
        return op.isdir(op.join(self.directory, val))

    def _is_valid_files(self, val):
        return val.endswith(self.extension)

    def get_new_contents(self):
        """Gets entire contents of directory and returns paths that were not
           present since last update
        """
        try:
            contents = set([i for i in os.listdir(self.directory)
                            if self._is_valid(i)])
        except OSError:
            contents = set()

        if len(contents) > len(self.directory_contents):
            new_contents = set(contents) - self.directory_contents
        else:
            new_contents = set()

        return new_contents

    def yield_new_paths(self):
        while True:
            current_modtime = op.getmtime(self.directory)
            if current_modtime > self.last_modtime:
                new_paths = self.get_new_contents()
                self.last_modtime = current_modtime
                if len(new_paths) > 0:
                    time.sleep(0.1)
                    self.directory_contents.update(new_paths)
                    for new_path in new_paths:
                        yield op.join(self.directory, new_path)
            else:
                time.sleep(0.1)


def collect_volumes_inotify(verbose=True):
    watch_manager = pyinotify.WatchManager()  # Watch Manager
    handler = DirectoryMonitorInotify(verbose=verbose)
    notifier = pyinotify.Notifier(watch_manager, handler)
    mask = pyinotify.IN_CLOSE_WRITE
    print('Watching {}'.format(config.SCANNER_DIR))
    watch_manager.add_watch(config.SCANNER_DIR, mask, auto_add=True)

    notifier.loop()


class DirectoryMonitorInotify(pyinotify.ProcessEvent):
    def __init__(self, extension='.dcm', verbose=True):
        self.redis_client = redis.StrictRedis(config.REDIS_HOST)
        self.extension = extension
        self.verbose = verbose

    def process_IN_CLOSE_WRITE(self, event):
        if (not op.isdir(event.pathname)) and (op.splitext(event.pathname)[1] == self.extension):
            self.redis_client.publish('volume', event.pathname)
            if self.verbose:
                print("Detected:", event.pathname)


def simulate_volumes(dataset):
    dest_directory = op.join(config.SCANNER_DIR, str(uuid4()))
    os.makedirs(dest_directory)
    paths = config.get_dataset_volume_paths(dataset)
    for i, path in enumerate(itertools.cycle(paths)):
        dest_path = op.join(dest_directory, f"IM{i:04}.dcm")
        shutil.copy(path, dest_path)
        yield path

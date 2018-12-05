#!/usr/bin/python
import os
import os.path as op
import re
import subprocess
import time
import logging

logging.basicConfig()
logger = logging.getLogger('trigger_inotify')


def collect_volumes_poll(directory=None, parent_directory=None, extension=None):
    """Continuously read the monitor directory and add publish new files as they 
    are detected.
    """
    if (directory is None) and (parent_directory is None):
        raise ValueError("Must provide either directory or parent_directory.")

    if parent_directory:
        monitor = DirectoryMonitorPoll(parent_directory, extension='/')
        logger.info('Detecting next subfolder in %s' % parent_directory)
        directory = op.join(parent_directory, next(monitor.yield_new_paths()))
        logger.info('Detected %s' % directory)

    monitor = DirectoryMonitorPoll(directory, extension=extension)

    for new_path in monitor.yield_new_paths():
        logger.info('Volume %s' % new_path)
        with open(new_path, 'rb') as f:
            pass


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
        samba_status = SambaStatus(directory)

        if extension == '/':
            self._is_valid = self._is_valid_directories
        else:
            self._is_valid = self._is_valid_files

        self.directory = directory
        self.extension = extension
        self.directory_contents = set()
        self.directory_contents = self.get_new_contents()
        self.samba_status = samba_status
        self.last_modtime = 0

    def _is_valid_directories(self, val):
        return op.isdir(op.join(self.directory, val))

    def _is_valid_files(self, val):
        if not val.endswith(self.extension):
            return False

        if val in self.samba_status.get_open_files():
            return False

        return True

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
                    self.directory_contents.update(new_paths)
                    for new_path in new_paths:
                        yield op.join(self.directory, new_path)
            else:
                time.sleep(0.05)


class SambaStatus(object):
    def __init__(self, directory):
        self.directory = directory
        self.open_file_parser = re.compile(b"\d*\s*\d*\s*[A-Z_]*\s*0x\d*\s*[A-Z]*\s*[A-Z]*\s*"
                                           b"%s\s*(?P<path>.*\.dcm).*" % directory)

    def get_open_files(self):
        proc = subprocess.Popen(['smbstatus', '-L'], stdout=subprocess.PIPE)
        proc.stdout.readline()
        proc.stdout.readline()
        proc.stdout.readline()

        paths = []
        for info in proc.stdout.readlines():
            if info != b'\n':
                groups = self.open_file_parser.match(info)
                if groups is not None:
                    path = groups.groupdict()['path']
                    paths.append(op.join(self.directory, path))

        return paths


if __name__ == "__main__":
    collect_volumes_poll(parent_directory='/mnt/scanner', extension='.dcm')

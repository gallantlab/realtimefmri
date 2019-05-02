#!/usr/bin/python
import logging
import os
import os.path as op
import pwd
import re
import subprocess
import time
from collections import defaultdict

import redis

# SETUP LOGGING
logger = logging.getLogger('samba.detect_dicoms')
logger.setLevel(logging.DEBUG)
LOG_FORMAT = '%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s'
formatter = logging.Formatter(LOG_FORMAT)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
fh = logging.FileHandler('/logs/samba.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

user = pwd.getpwuid(os.getuid()).pw_name
logger.info("Running as user %s", user)


def detect_dicoms(root_directory=None, extension='*'):
    """Continuously monitor a samba mounted directory for new files and publish new paths.

    File creation on samba network shares do not trigger the same inotify events as regular files.
    This function monitors a samba shared directory for new files. When a new file is detected,
    ensure it is closed, then publish the name over redis.

    Parameters
    ----------
    directory : str
        Directory to monitor for new files
    extension : str
        Only detect new files with this extension
    """
    logger.info('Monitoring %s', root_directory)

    monitor = MonitorSambaDirectory(root_directory, extension=extension)

    r = redis.StrictRedis('redis')

    for new_path in monitor.yield_new_paths():
        new_path = new_path.replace(root_directory, '', 1).lstrip('/')
        logger.info(f'SAMBA got a new volume {new_path} at time {time.time()}')
        r.publish('volume', new_path)


class MonitorSambaDirectory(object):
    """
    Monitor the file contents of a directory mounted with samba share

    Parameters
    ----------
    directory : str
        The directory to monitor
    extension : str

    Examples
    --------
    Loop that iterates each time a file is detected.

    >>> m = MonitorSambaDirectory('/tmp/test', extension='.dcm')
    >>> for path in m.yield_new_paths():
    >>>     print(path)
    /tmp/test/1.dcm
    /tmp/test/2.dcm
    ...
    """
    def __init__(self, root_directory, extension='.dcm'):
        samba_status = SambaStatus(root_directory)

        self.root_directory = root_directory
        self.directories = set(self.get_directories())
        self.extension = extension
        self.samba_status = samba_status

        self.last_modtimes = {}
        self.directory_contents = defaultdict(set)


        for directory in self.directories:
            self.last_modtimes[directory] = 0
            added_contents, _ = self.get_changed_contents(directory, set(), self.is_valid_file)
            self.directory_contents[directory] = added_contents

    def get_directories(self):
        return [p for p in os.listdir(self.root_directory)
                if op.isdir(op.join(self.root_directory, p))]

    def is_valid_directory(self, val):
        return op.isdir(op.join(self.root_directory, val))

    def is_valid_file(self, val):
        if not val.endswith(self.extension):
            return False

        if val in self.samba_status.get_open_files():
            return False

        return True

    def get_changed_contents(self, directory, previous_contents, validator):
        """Gets entire contents of directory and returns paths that were not
        present since last update
        """
        current_contents = set([op.basename(p)
                                for p in os.listdir(op.join(self.root_directory, directory))
                                if validator(p)])

        if len(current_contents) > len(previous_contents):
            added_contents = current_contents - previous_contents
            removed_contents = set()
        elif len(current_contents) < len(previous_contents):
            added_contents = set()
            removed_contents = previous_contents - current_contents
        else:
            added_contents, removed_contents = set(), set()

        return added_contents, removed_contents

    def yield_new_paths(self):
        while True:

            # any new directories?
            (added_directories,
             removed_directories) = self.get_changed_contents('', self.directories,
                                                              self.is_valid_directory)
            if len(added_directories) > 0:
                for directory in added_directories:
                    logger.info('Adding directory %s', directory)
                    self.last_modtimes[directory] = 0
                    self.directories.add(directory)

            if len(removed_directories) > 0:
                for directory in removed_directories:
                    logger.info('Removing directory %s', directory)
                    del self.last_modtimes[directory]
                    self.directories.remove(directory)

            for directory in self.directories:
                current_modtime = op.getmtime(op.join(self.root_directory, directory))
                if current_modtime > self.last_modtimes[directory]:
                    logger.info('Detected change in %s', directory)

                    (added_paths,
                     removed_paths) = self.get_changed_contents(directory,
                                                                self.directory_contents[directory],
                                                                self.is_valid_file)

                    self.last_modtimes[directory] = current_modtime

                    def get_mod_time(filename):
                        return op.getmtime(op.join(self.root_directory,
                                                   directory, filename))

                    if len(added_paths) > 0:
                        for path in sorted(added_paths, key=get_mod_time):
                            path = op.basename(path)
                            logger.info('Adding %s from %s', path, directory)
                            self.directory_contents[directory].add(path)
                            yield op.join(self.root_directory, directory, path)

                    if len(removed_paths) > 0:
                        for path in removed_paths:
                            path = op.basename(path)
                            logger.info('Removing %s from %s', path, directory)
                            self.directory_contents[directory].remove(path)

            time.sleep(0.1)


class SambaStatus():
    """Class to access information output by the `smbstatus` command.

    Parameters
    ----------
    directory : str
        Only return information related to this directory
    """
    def __init__(self, directory):
        self.directory = directory
        self.open_file_parser = re.compile("\d*\s*\d*\s*[A-Z_]*\s*0x\d*\s*[A-Z]*\s*[A-Z]*\s*"
                                           "%s\s*(?P<path>.*\.dcm).*" % directory)

    def get_open_files(self):
        """Get a list of files that are currently opened by samba clients
        """
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
    detect_dicoms(root_directory='/mnt/scanner', extension='.dcm')

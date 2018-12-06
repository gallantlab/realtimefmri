#!/usr/bin/python
import os
import os.path as op
import pwd
import re
import subprocess
from collections import defaultdict
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trigger_inotify')

user = pwd.getpwuid(os.getuid()).pw_name
logger.info("Running as user {}".format(user))


def trigger_inotify(root_directory=None, parent_directory=None, extension='*'):
    """Continuously monitor a samba mounted directory for new files and trigger inotify actions.

    File creation on samba network shares do not trigger the same inotify events as regular files. 
    This function monitors a samba shared directory for new files. When a new file is detected, 
    open and close it to result in inotify events being triggered.

    Parameters
    ----------
    directory : str
        Directory to monitor for new files
    parent_directory : str
        If provided, monitor the first new directory created in this parent directory
    extension : str
        Only detect new files with this extension
    """
    logger.info('Monitoring %s' % root_directory)

    monitor = MonitorSambaDirectory(root_directory, extension=extension)

    for new_path in monitor.yield_new_paths():
        logger.info('Volume %s' % new_path)
        with open(new_path, 'rb') as f:
            pass


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
                    logging.info('Adding directory {}'.format(directory))
                    self.last_modtimes[directory] = 0
                    self.directories.add(directory)

            if len(removed_directories) > 0:
                for directory in removed_directories:
                    logging.info('Removing directory {}'.format(directory))
                    del self.last_modtimes[directory]
                    self.directories.remove(directory)

            for directory in self.directories:
                current_modtime = op.getmtime(op.join(self.root_directory, directory))
                if current_modtime > self.last_modtimes[directory]:
                    logging.info('Detected change in {}'.format(directory))

                    (added_paths,
                     removed_paths) = self.get_changed_contents(directory,
                                                                self.directory_contents[directory],
                                                                self.is_valid_file)

                    self.last_modtimes[directory] = current_modtime
                    if len(added_paths) > 0:
                        for path in added_paths:
                            path = op.basename(path)
                            logging.info('Adding {} from {}'.format(path, directory))
                            self.directory_contents[directory].add(path)
                            yield op.join(self.root_directory, directory, path)

                    if len(removed_paths) > 0:
                        for path in removed_paths:
                            path = op.basename(path)
                            logging.info('Removing {} from {}'.format(path, directory))
                            self.directory_contents[directory].remove(path)

            time.sleep(0.1)


class SambaStatus(object):
    """Class to access information output by the `smbstatus` command.

    Parameters
    ----------
    directory : str
        Only return information related to this directory
    """
    def __init__(self, directory):
        self.directory = directory
        self.open_file_parser = re.compile(b"\d*\s*\d*\s*[A-Z_]*\s*0x\d*\s*[A-Z]*\s*[A-Z]*\s*"
                                           b"%s\s*(?P<path>.*\.dcm).*" % directory)

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
    trigger_inotify(root_directory='/mnt/scanner', extension='.dcm')

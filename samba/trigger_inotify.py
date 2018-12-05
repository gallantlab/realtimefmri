#!/usr/bin/python
import os
import os.path as op
import pwd
import re
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trigger_inotify')

user = pwd.getpwuid(os.getuid()).pw_name
logger.info("Running as user {}".format(user))


def trigger_inotify(directory=None, parent_directory=None, extension='*'):
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
    if (directory is None) and (parent_directory is None):
        raise ValueError("Must provide either directory or parent_directory.")

    if parent_directory:
        monitor = MonitorSambaDirectory(parent_directory, extension='/')
        logger.info('Detecting next subfolder in %s' % parent_directory)
        directory = op.join(parent_directory, next(monitor.yield_new_paths()))
        logger.info('Detected %s' % directory)

    monitor = MonitorSambaDirectory(directory, extension=extension)

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
    def __init__(self, directory, extension='.dcm'):
        samba_status = SambaStatus(directory)

        if extension == '/':
            self._is_valid = self._is_valid_directories
        elif extension == '*':
            self._is_valid = lambda p: True
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
    trigger_inotify(parent_directory='/mnt/scanner', extension='.dcm')

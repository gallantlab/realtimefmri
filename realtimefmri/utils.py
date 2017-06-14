'''
Utility functions and configuration
'''
import six
if six.PY2:
    input = raw_input

import os.path as op
from glob import glob
import struct

import logging
import logging.handlers
import tempfile

import numpy as np
from nibabel import Nifti1Image, load as nibload

from realtimefmri.config import LOG_LEVEL, LOG_FORMAT, RECORDING_DIR


def parse_message(message):
    topic, sync_time, data = message
    topic = topic.decode('utf8')
    sync_time = struct.unpack('d', sync_time)[0]
    return topic, sync_time, data


def load_run(recording_id):
    """Load data from a real-time run into a nifti volumes
    """
    file_paths = glob(op.join(RECORDING_DIR, recording_id, '*.nii'))
    file_paths = sorted(file_paths)
    volume = None
    for i, file_path in enumerate(file_paths):
        nii = nibload(file_path)

        if volume is None:
            x, y, z = nii.shape
            shape = (len(file_paths), x, y, z)
            volume = np.zeros(shape)
            affine = nii.affine

        assert nii.affine == affine

        volume[i, ...] = nii.get_data()

    return Nifti1Image(volume, affine)


def get_temporary_file_name(root=tempfile.gettempdir()):
    from random import choice
    from string import ascii_letters, digits

    done = False
    characters = ascii_letters + digits
    while not done:

        unique_part = ''.join([choice(characters) for i in range(10)])
        temp_fname = op.join(root, tempfile.gettempprefix() + unique_part)
        if not op.exists(temp_fname):
            done = True

    return temp_fname


def confirm(prompt, choices=('y', 'n')):
    '''
    Get user input from a list of possibilities
    '''
    choice = None
    while choice not in choices:
        choice = input(prompt+'> ')
    return choice


def get_logger(name, to_console=False, to_file=False, to_network=False,
               level=LOG_LEVEL, formatting=LOG_FORMAT):
    '''
    Returns a logger to some desired destinations. Checks to see if the logger
    or any of its parents already have the requested destinations to avoid
    duplicate log entries

    Args:
        name (str): the name of the logger
                    can indicate logger hierarchy, e.g. calling:
                        lg1 = get_logger('test')
                        lg2 = get_logger('test.2')
                    produces two loggers. lg2 logs to its own destinations as
                    well as the destinations specified in lg1

        to_console (bool): attaches a stream handler that logs to the console
                           level=INFO)
        to_file (bool or str): True, attaches a file handler that logs to a
                               file in the logging directory any other string
                               logs to that (absolute) path
        to_network (bool or int): True, sends log to network destination at
                                  default TCP port int, sends to a specific
                                  TCP port

    Example:
        lg1 = get_logger('test', to_file=True, to_console=True)
        lg2 = get_logger('test.2', to_file=True)

        # logs "wefwef" to a file "##datetime##_test.log" in the logging
        # directory
        lg1.debug('wefwef')

        # logs "hibye" to the lg1 file and ##datetime##_test.2.log in the
        # logging directory
        lg2.debug('hibye')
    '''
    formatter = logging.Formatter(formatting)

    if to_network:
        to_network = logging.handlers.DEFAULT_TCP_LOGGING_PORT

    def has_stream_handler(logger):
        '''Checks if this logger or any of its parents has a stream handler.
           Short circuits if it finds a stream handler at any level'''
        has = any([isinstance(h, logging.StreamHandler)
                   for h in logger.handlers])
        if has:
            return True
        elif ((logger.parent is not None) and
              (not isinstance(logger.parent, logging.RootLogger))):
            return has_stream_handler(logger.parent)
        else:
            return False

    def has_file_handler(logger, fname):
        '''Checks if this logger or any of its parents has a file handler.
        '''
        has = any([h.baseFilename == fname for h in logger.handlers
                   if isinstance(h, logging.FileHandler)])
        if has:
            return True
        elif ((logger.parent is not None) and
              (not isinstance(logger.parent, logging.RootLogger))):
            return has_file_handler(logger.parent, fname)
        else:
            return False

    def has_network_handler(logger, port):
        return False
        # has = any([h.baseFilename==fname for h in logger.handlers
        #            if type(h) is logging.FileHandler])
        # if has==True:
        #     return True
        # elif type(logger.parent) is not logging.RootLogger:
        #     return has_file_handler(logger.parent, fname)
        # else:
        #     return False

    if name == 'root':
        logger = logging.root
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    if to_console:
        if not has_stream_handler(logger):
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(level)
            logger.addHandler(ch)

    if to_file and not has_file_handler(logger, to_file):
        if op.exists(to_file):
            msg = ('File {} exists. '
                   'Append/Overwrite/Cancel (a/o/c)?'.format(to_file))
            choice = confirm(msg, choices=('a', 'o', 'c'))
            if choice.lower() == 'a':
                mode = 'a'
            elif choice.lower() == 'o':
                mode = 'w'
            elif choice.lower() == 'c':
                raise IOError('Log for {} exists'.format(to_file))
        else:
            mode = 'w'
        fh = logging.FileHandler(to_file, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if to_network and not has_network_handler(logger, to_network):
        nh = logging.handlers.SocketHandler('localhost', to_network)
        nh.setFormatter(formatter)
        logger.addHandler(nh)

    return logger

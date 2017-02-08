import os
import time
import logging
import struct
import zmq

class LocalTimedObject(object):
    def __init__(self, *args, **kwargs):
        pass
    @property
    def timestamp(self):
        return time.time()

class NetworkTimedObject(object):
    def __init__(self, port=5550, **kwargs):
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect('tcp://localhost:{}'.format(port))
    @property
    def timestamp(self):
        self.socket.send('')
        return struct.unpack('d', self.socket.recv())[0]

# package_directory = '/home/glab/Documents/realtimefmri'
package_directory = '/auto/k1/robertg/code/realtimefmri'
log_directory = os.path.join(package_directory, 'log')

database_directory = os.path.join(package_directory, 'database')

def get_subject_directory(subject):
    return os.path.join(database_directory, subject)

def get_example_data_directory(dataset):
    return os.path.join(package_directory, 'datasets', dataset)

def confirm(prompt, choices=['y','n']):
    choice = None
    while not choice in choices:
        choice = raw_input(prompt+'> ')
    return choice

log_directory = os.path.join(package_directory, 'log')
test_data_directory = os.path.join(package_directory, 'tests/data')
recording_directory = os.path.join(package_directory, 'recordings')
configuration_directory = os.path.join(package_directory, 'config')

def get_logger(name, dest=[], level=logging.DEBUG):
    '''
    Returns a logger to some desired destinations. Checks to see if the logger or any of its parents
    already have the requested destinations to avoid duplicate log entries

    Input
    ------
    name (str):    the name of the logger
                can indicate logger hierarchy, e.g. calling:
                    lg1 = get_logger('test')
                    lg2 = get_logger('test.2')
                produces two loggers. lg2 logs to its own destinations as well as the
                destinations specified in lg1

    dest (list): a list of logging destinations
                - 'console' attaches a stream handler that logs to the console (level=INFO)
                - 'file' attaches a file handler that logs to a file in the logging directory
                - any other string logs to that (absolute) path
    Example
    ------
        lg1 = get_logger('test', dest=['file', 'console'])
        lg2 = get_logger('test.2', dest='file')

        lg1.debug('wefwef') # logs "wefwef" to a file "##datetime##_test.log" in the logging directory
        lg2.debug('hibye') # logs "hibye" to the lg1 file and ##datetime##_test.2.log in the logging directory
    '''
    def has_stream_handler(logger):
        '''Checks if this logger or any of its parents has a stream handler. Short circuits if it finds a
        stream handler at any level'''
        has = any([type(h) is logging.StreamHandler for h in logger.handlers])
        if has==True:
            return True
        elif type(logger.parent) is not logging.RootLogger:
            return has_stream_handler(logger.parent)
        else:
            return False

    def has_file_handler(logger, fname):
        has = any([h.baseFilename==fname for h in logger.handlers
                   if type(h) is logging.FileHandler])
        if has==True:
            return True
        elif type(logger.parent) is not logging.RootLogger:
            return has_file_handler(logger.parent, fname)
        else:
            return False

    if not type(dest) in (list, tuple):
        dest = [dest]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if 'console' in dest:
        if not has_stream_handler(logger):
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            logger.addHandler(ch)
        dest.remove('console')
    for d in dest:
        if d=='file':
            log_name = '%s_%s.log' % (time.strftime('%Y%m%d'), name)
            log_path = os.path.join(log_directory, log_name)
        else:
            log_path = d

        if not has_file_handler(logger, log_path):
            if os.path.exists(log_path):
                choice = confirm('File {} exists. Append/Overwrite/Cancel (a/o/c)?'.format(log_path), choices=('a','o','c'))
                if choice.lower()=='a':
                    mode = 'a'
                elif choice.lower()=='o':
                    mode = 'w'
                elif choice.lower()=='c':
                    raise IOError, 'Log for {} exists'.format(log_path)
            else:
                mode = 'w'
            formatter = logging.Formatter('%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s')
            fh = logging.FileHandler(log_path, mode=mode)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def generate_command(command, params):
    cmd = [command]

    kw_args = [i for i in params if ('flag' in i) and ('value' in i)]
    first_pos_args = [i for i in params if i.get('position', None)=='first']
    last_pos_args = [i for i in params if i.get('position', None)=='last']
    flag_args = [i for i in params if not 'value' in i]

    for a in first_pos_args:
        cmd.append(a['value'])

    for a in kw_args:
        cmd.extend(['-'+a['flag'], a['value']])

    for a in flag_args:
        cmd.append('-'+a['flag'])

    for a in last_pos_args:
        cmd.append(a['value'])

    return cmd

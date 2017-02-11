import os.path as op
from time import strftime
import logging, logging.handlers

# package_directory = '/home/glab/Documents/realtimefmri'
package_directory = '/auto/k1/robertg/code/realtimefmri'
log_directory = op.join(package_directory, 'log')

database_directory = op.join(package_directory, 'database')

def get_subject_directory(subject):
    return op.join(database_directory, subject)

def get_example_data_directory(dataset):
    return op.join(package_directory, 'datasets', dataset)

def confirm(prompt, choices=['y','n']):
    choice = None
    while not choice in choices:
        choice = raw_input(prompt+'> ')
    return choice

log_directory = op.join(package_directory, 'log')
test_data_directory = op.join(package_directory, 'tests/data')
recording_directory = op.join(package_directory, 'recordings')
configuration_directory = op.join(package_directory, 'config')

def get_logger(name, to_console=False, to_file=False, to_network=False, level=logging.INFO,
               formatting='%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s'):
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

    to_console (bool): attaches a stream handler that logs to the console (level=INFO)
    to_file (bool or str): True, attaches a file handler that logs to a file in the logging directory
                           any other string logs to that (absolute) path
    to_network (bool or int): True, sends log to network destination at default TCP port
                              int, sends to a specific TCP port

    Example
    ------
        lg1 = get_logger('test', to_file=True, to_console=True)
        lg2 = get_logger('test.2', to_file=True)

        lg1.debug('wefwef') # logs "wefwef" to a file "##datetime##_test.log" in the logging directory
        lg2.debug('hibye') # logs "hibye" to the lg1 file and ##datetime##_test.2.log in the logging directory
    '''
    formatter = logging.Formatter(formatting)

    if to_file==True:
        log_name = '%s_%s.log' % (strftime('%Y%m%d'), name)
        to_file = op.join(log_directory, log_name)
    if to_network==True:
        to_network = logging.handlers.DEFAULT_TCP_LOGGING_PORT


    def has_stream_handler(logger):
        '''Checks if this logger or any of its parents has a stream handler. Short circuits if it finds a
        stream handler at any level'''
        has = any([type(h) is logging.StreamHandler for h in logger.handlers])
        if has==True:
            return True
        elif ((logger.parent is not None) and 
             (type(logger.parent) is not logging.RootLogger)):
            return has_stream_handler(logger.parent)
        else:
            return False

    def has_file_handler(logger, fname):
        has = any([h.baseFilename==fname for h in logger.handlers
                   if type(h) is logging.FileHandler])
        if has==True:
            return True
        elif ((logger.parent is not None) and 
             (type(logger.parent) is not logging.RootLogger)):
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

    if name=='root':
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
            choice = confirm(msg, choices=('a','o','c'))
            if choice.lower()=='a':
                mode = 'a'
            elif choice.lower()=='o':
                mode = 'w'
            elif choice.lower()=='c':
                raise IOError, 'Log for {} exists'.format(to_file)
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
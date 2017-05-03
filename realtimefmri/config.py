import os.path as op
import logging
import realtimefmri

# PATHS
# PACKAGE_DIR = '/home/glab/code/realtimefmri'
PACKAGE_DIR = '/auto/k1/robertg/code/realtimefmri'

DATABASE_DIR = op.join(PACKAGE_DIR, 'database')
TEST_DATA_DIR = op.join(PACKAGE_DIR, 'tests/data')
RECORDING_DIR = op.join(PACKAGE_DIR, 'recordings')
PIPELINE_DIR = op.join(PACKAGE_DIR, 'pipelines')
SCRIPTS_DIR = op.join(PACKAGE_DIR, 'scripts')
MODULE_DIR = realtimefmri.__path__[0]

# PORTS
SYNC_PORT = 5556
VOLUME_PORT = 5557
PREPROC_PORT = 5558
STIM_PORT = 5559

# TTL
KEYBOARD_FN = '/dev/input/event2'
TTL_SERIAL_PORT = '/dev/ttyUSB0'

# LOGGING
LOG_FORMAT = '%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s'
LOG_LEVEL = logging.INFO


def get_subject_directory(subject):
    '''Subject directory'''
    return op.join(DATABASE_DIR, subject)


def get_example_data_directory(dataset):
    '''Example data directory'''
    return op.join(PACKAGE_DIR, 'datasets', dataset)

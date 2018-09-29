#!/usr/bin/env python3
import os.path as op
import logging
from configparser import ConfigParser
import realtimefmri


CONFIG_DIR = op.expanduser('~/.config/realtimefmri')

config = ConfigParser()
config.read(op.join(CONFIG_DIR, 'config.cfg'))

DATA_DIR = op.expanduser(config.get('directories', 'data'))
SCANNER_DIR = op.expanduser(config.get('directories', 'scanner'))
DATABASE_DIR = op.expanduser(config.get('directories', 'database'))
RECORDING_DIR = op.expanduser(config.get('directories', 'recordings'))
PIPELINE_DIR = op.expanduser(config.get('directories', 'pipelines'))
DATASET_DIR = op.expanduser(config.get('directories', 'datasets'))
MODULE_DIR = realtimefmri.__path__[0]

# PORTS
SYNC_PORT = int(config.get('ports', 'sync'))
VOLUME_PORT = int(config.get('ports', 'volume'))
PREPROC_PORT = int(config.get('ports', 'preproc'))
STIM_PORT = int(config.get('ports', 'stim'))

# TTL
KEYBOARD_FN = config.get('sync', 'keyboard')
TTL_SERIAL_PORT = config.get('sync', 'serial')

# LOGGING
LOG_FORMAT = '%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s'
LOG_LEVEL = logging.INFO


def get_example_data_directory(dataset):
    '''Example data directory'''
    return op.join(DATA_DIR, 'datasets', dataset)


def get_subject_directory(subject):
    return op.join(DATABASE_DIR, subject)

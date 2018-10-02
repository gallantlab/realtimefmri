#!/usr/bin/env python3
import os
import os.path as op
from glob import glob
import shutil
import logging
from configparser import ConfigParser
import realtimefmri


def initialize_config():
    if not op.exists(op.join(CONFIG_DIR, 'config.cfg')):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        shutil.copy(op.join(PACKAGE_DIR, 'config.cfg'), CONFIG_DIR)


def initialize():
    if not op.exists(PIPELINE_DIR):
        os.makedirs(PIPELINE_DIR)

        for path in glob(op.join(CONFIG_DIR, 'pipelines', '*.yaml')):
            shutil.copy(path, PIPELINE_DIR)

    if not op.exists(SCANNER_DIR):
        os.makedirs(SCANNER_DIR)

    if not op.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    if not op.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)

    if not op.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    if not op.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)


def get_example_data_directory(dataset):
    """Return the directory of sample data"""
    return op.join(DATASET_DIR, dataset)


def get_subject_directory(subject):
    return op.join(DATABASE_DIR, subject)


PACKAGE_DIR = op.abspath(op.join(realtimefmri.__file__, op.pardir))
CONFIG_DIR = op.expanduser('~/.config/realtimefmri')
initialize_config()

config = ConfigParser()
config.read(op.join(CONFIG_DIR, 'config.cfg'))

DATA_DIR = op.expanduser(config.get('directories', 'data'))
PIPELINE_DIR = op.expanduser(config.get('directories', 'pipelines'))
SCANNER_DIR = op.expanduser(config.get('directories', 'scanner'))
DATABASE_DIR = op.expanduser(config.get('directories', 'database'))
RECORDING_DIR = op.expanduser(config.get('directories', 'recordings'))
DATASET_DIR = op.expanduser(config.get('directories', 'datasets'))
initialize()

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

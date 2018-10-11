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

        for path in glob(op.join(PACKAGE_DIR, 'pipelines', '*.yaml')):
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


def get_dataset(dataset):
    """Return the directory of a dataset"""
    return op.join(DATASET_DIR, dataset)


def get_datasets():
    """List all available datasets"""
    paths = sorted([d for d in os.listdir(DATASET_DIR)
                    if op.isdir(op.join(DATASET_DIR, d))])
    datasets = []
    for path in paths:
        datasets.append(op.splitext(op.basename(path))[0])

    return datasets


def get_dataset_volume_paths(dataset, extension='.dcm'):
    directory = get_dataset(dataset)
    return sorted(glob(op.join(directory, '*' + extension)))


def get_pipelines(pipeline_type):
    paths = sorted(glob(op.join(PIPELINE_DIR, pipeline_type + '-*.yaml')))
    pipelines = []
    for path in paths:
        pipelines.append(op.splitext(op.basename(path))[0])

    return pipelines


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

# addresses
SYNC_ADDRESS = config.get('addresses', 'sync')
VOLUME_ADDRESS = config.get('addresses', 'volume')
PREPROC_ADDRESS = config.get('addresses', 'preproc')
STIM_ADDRESS = config.get('addresses', 'stim')
REDIS_HOST = config.get('addresses', 'redis_host')

# TTL
TTL_KEYBOARD_DEV = config.get('sync', 'keyboard')
TTL_SERIAL_DEV = config.get('sync', 'serial')
TTL_ZMQ_ADDR = config.get('sync', 'zmq')

# LOGGING
LOG_FORMAT = '%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s'
LOG_LEVEL = logging.INFO

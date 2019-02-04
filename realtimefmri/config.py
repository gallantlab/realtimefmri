#!/usr/bin/env python3
import logging
import os
import os.path as op
import shutil
from configparser import ConfigParser
from glob import glob

import cortex
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

    if not op.exists(DATASTORE_DIR):
        os.makedirs(DATASTORE_DIR)

    if not op.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)

    if not op.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)


def get_surfaces():
    return sorted(list(cortex.db.subjects.keys()))


def get_available_transforms(surface):
    """Get available pycortex transforms for a surface"""
    try:
        surf = getattr(cortex.db, surface)
        transforms = sorted(surf.transforms.xfms)
    except AttributeError:
        transforms = []

    return transforms


def get_available_masks(surface, transform):
    """Get available pycortex transforms for a surface"""
    try:
        surf = getattr(cortex.db, surface)
        transf = surf.transforms[transform]
        masks = transf.masks._masks.keys()
        masks = sorted(list(masks))

    except AttributeError:
        masks = []

    return masks


def get_dataset(dataset):
    """Return the directory of a dataset"""
    return op.join(DATASET_DIR, dataset)


def get_datasets():
    """List all available datasets"""
    paths = sorted([d for d in os.listdir(DATASET_DIR)
                    if op.isdir(op.join(DATASET_DIR, d))])
    datasets = []
    for path in paths:
        datasets.append(op.basename(path))

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
    return op.join(DATASTORE_DIR, subject)


PACKAGE_DIR = op.abspath(op.join(realtimefmri.__file__, op.pardir))

if os.geteuid() == 0:
    CONFIG_DIR = '/etc/realtimefmri'
    DATA_DIR = '/usr/local/share/realtimefmri'
else:
    CONFIG_DIR = op.expanduser('~/.config/realtimefmri')
    DATA_DIR = op.expanduser('~/.local/share/realtimefmri')

initialize_config()

PIPELINE_DIR = op.join(DATA_DIR, 'pipelines')
EXPERIMENT_DIR = op.join(DATA_DIR, 'experiments')
SCANNER_DIR = op.join(DATA_DIR, 'scanner')
DATASTORE_DIR = op.join(DATA_DIR, 'datastore')
RECORDING_DIR = op.join(DATA_DIR, 'recordings')
DATASET_DIR = op.join(DATA_DIR, 'datasets')
initialize()

config = ConfigParser()
config.read(op.join(CONFIG_DIR, 'config.cfg'))

# addresses
REDIS_HOST = config.get('addresses', 'redis_host')

# web
STATIC_PATH = config.get('web', 'static')

# TTL
TTL_KEYBOARD_DEV = config.get('sync', 'keyboard')
TTL_SERIAL_DEV = config.get('sync', 'serial')

# LOGGING
LOG_FORMAT = '%(asctime)-12s %(name)-20s %(levelname)-8s %(message)s'
log_level_name = os.getenv('REALTIMEFMRI_LOG_LEVEL', 'INFO')
LOG_LEVEL = logging.getLevelName(log_level_name)

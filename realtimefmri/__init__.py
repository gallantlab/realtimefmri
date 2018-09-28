import os
from realtimefmri import config


def makedirs_except(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


def initialize():
    makedirs_except(config.SCANNER_DIR)
    makedirs_except(config.RECORDING_DIR)
    makedirs_except(config.DATABASE_DIR)
    makedirs_except(config.PIPELINE_DIR)
    makedirs_except(config.DATASET_DIR)

initialize()

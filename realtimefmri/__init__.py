import os
from realtimefmri import config


def makedirs_except(directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass

def initialize():
    makedirs_except(config.DATABASE_DIR)
    makedirs_except(config.RECORDING_DIR)
    makedirs_except(config.PIPELINE_DIR)
    makedirs_except(config.DATASET_DIR)

initialize()

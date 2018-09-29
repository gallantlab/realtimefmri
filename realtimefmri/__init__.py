import os
from realtimefmri import config


def initialize():
    os.makedirs(config.SCANNER_DIR, exist_ok=True)
    os.makedirs(config.RECORDING_DIR, exist_ok=True)
    os.makedirs(config.DATABASE_DIR, exist_ok=True)
    os.makedirs(config.PIPELINE_DIR, exist_ok=True)
    os.makedirs(config.DATASET_DIR, exist_ok=True)


initialize()

import os
import os.path as op
from glob import glob
import shutil
import argparse
from realtimefmri.config import get_example_data_directory
from uuid import uuid4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('simulate_dataset', action='store')
    args = parser.parse_args()

    ex_directory = get_example_data_directory(args.simulate_dataset)
    paths = glob(op.join(ex_directory, '*.PixelData'))

    dest_directory = op.join('/tmp/rtfmri', str(uuid4()))
    print('Simulated volumes appearing in {}'.format(dest_directory))

    os.makedirs(dest_directory)

    for path in paths:
        input('>>> press 5 for TTL, then enter for new image')
        new_path = op.join(dest_directory, str(uuid4())+'.PixelData')
        shutil.copy(path, new_path)

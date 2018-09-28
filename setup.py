"""Setup file for realtimefmri"""
import os.path as op
from setuptools import setup, find_packages

CONFIG_DIR = op.expanduser('~/.config/realtimefmri')
PIPELINE_DIR = op.join(CONFIG_DIR, 'pipelines')


def main():
    """Main setup function"""
    setup(name='realtimefmri',
          version='0.1.1',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=find_packages(),
          include_package_data=True,
          install_requires=["Cython",
                            "numpy",
                            "scipy",
                            "numexpr",
                            "h5py",
                            "matplotlib",
                            "nibabel",
                            "Pillow",
                            "pyparsing",
                            "pyserial",
                            "PyYAML",
                            "pyzmq",
                            "six",
                            "tornado",
                            "evdev"],

          data_files=[(CONFIG_DIR, ['config.cfg']),
                      (PIPELINE_DIR, ['pipelines/preproc-zscore.yaml',
                                      'pipelines/stim-debug.yaml',
                                      'pipelines/stim-viewer.yaml'])],

          entry_points={'console_scripts':
                        ['realtimefmri = realtimefmri.__main__:main']})


if __name__ == '__main__':
    main()

"""Setup file for realtimefmri"""
from setuptools import setup


def main():

    """Main setup function"""
    setup(name='realtimefmri',
          version='0.1.1',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=['realtimefmri'],
          install_requires=["numpy",
                            "matplotlib",
                            "pydicom",
                            "nibabel",
                            "dicom2nifti",
                            "pycortex",
                            "pyserial",
                            "PyYAML",
                            "pyzmq",
                            "six",
                            "tornado",
                            "evdev"],
          entry_points={'console_scripts':
                        ['realtimefmri = realtimefmri.__main__:main']},
          package_data={'realtimefmri': ['config.cfg', 'pipelines/*-debug.yaml']})


if __name__ == '__main__':
    main()

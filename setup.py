"""Setup file for realtimefmri"""
from setuptools import setup


def main():

    """Main setup function"""
    setup(name='realtimefmri',
          version='0.1.2',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=['realtimefmri'],
          install_requires=["numpy",
                            "redis",
                            "dash",
                            "dash_core_components",
                            "dash_html_components",
                            "matplotlib",
                            "pydicom",
                            "nibabel",
                            "dicom2nifti",
                            "pycortex",
                            "pyserial",
                            "evdev",
                            "PyYAML",
                            "pyzmq"],
          entry_points={'console_scripts':
                        ['realtimefmri = realtimefmri.__main__:main']},
          package_data={'realtimefmri': ['config.cfg', 'pipelines/*-debug.yaml']})


if __name__ == '__main__':
    main()

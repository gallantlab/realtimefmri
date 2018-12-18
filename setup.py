"""Setup file for realtimefmri"""
from setuptools import setup, find_packages


def main():

    """Main setup function"""
    setup(name='realtimefmri',
          version='0.1.2',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=find_packages(),
          install_requires=["numpy",
                            "redis",
                            "matplotlib",
                            "nibabel",
                            "pydicom",
                            "pycortex",
                            "dash",
                            "dash_core_components",
                            "dash_html_components",
                            "scikit-learn",
                            "pyinotify",
                            "pyserial",
                            "evdev",
                            "PyYAML"],
          entry_points={'console_scripts':
                        ['realtimefmri = realtimefmri.__main__:main']},
          package_data={'realtimefmri': ['config.cfg', 'pipelines/*-debug.yaml']})


if __name__ == '__main__':
    main()

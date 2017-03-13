"""Setup file for realtimefmri"""
from setuptools import setup, find_packages


def main():
    """Main setup function"""
    setup(name='realtimefmri',
          version='0.1.1',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=find_packages(),
          entry_points={'console_scripts':
                        ['realtimefmri = realtimefmri.__main__:main']})


if __name__ == '__main__':
    main()

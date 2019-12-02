"""Setup file for realtimefmri"""
from setuptools import find_packages, setup


def main():
    """Main setup function"""
    setup(name='realtimefmri',
          version='0.1.1.dev0',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=find_packages(),
          install_requires=["numpy",
                            "redis",
                            "nibabel",
                            "pydicom",
                            "pycortex",
                            "dash",
                            "dash_core_components",
                            "dash_html_components",
                            "scikit-learn",
                            "pyserial",
                            "evdev",
                            "PyYAML"],
          entry_points={'console_scripts':
                        ['realtimefmri = realtimefmri.__main__:main']},
          package_data={'realtimefmri': ['config.cfg', 'pipelines/preproc-default.yaml']})


if __name__ == '__main__':
    main()

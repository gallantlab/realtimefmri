from setuptools import setup, find_packages

def main():
    setup(name='realtimefmri',
          version='0.1.1',
          description='code for realtime fmri',
          author='robertg',
          author_email='robertg@berkeley.edu',
          packages=find_packages())

if __name__=='__main__':
    main()
    

'''Data collection code
'''
import os
import time
from glob import glob
from itertools import cycle
import zmq
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, FileSystemEventHandler

from .utils import get_example_data_directory, get_logger


class DataCollector(object):
    '''Connects to
    '''
    def __init__(self, out_port=5556, verbose=False):
        '''
        Arguments
        directory: string with directory to watch for files, or if
                   parent_directory is True, new folder
        parent_directory: bool indicating if provided directory is the image
                          directory or a parent directory to watch
        simulate: True indicates we should simulate from provided
                  directory
        interval: return, or int indicating seconds between simulated image
                  acquisition
        '''

        super(DataCollector, self).__init__()
        logger = get_logger('collecting', to_console=verbose, to_network=True)
        logger.debug('data collector initialized')

        context = zmq.Context()
        self.image_pub = context.socket(zmq.PUB)
        self.image_pub.bind('tcp://*:%d' % out_port)

        self.logger = logger

    def send_image(self, image_fpath):
        '''Load image from path and publish to subscribers
        '''
        with open(image_fpath, 'r') as f:
            raw_image_binary = f.read()

        self.logger.info('%s %u', os.path.basename(image_fpath),
                         len(raw_image_binary))

        self.image_pub.send_multipart([b'image', raw_image_binary])


class Simulator(DataCollector):
    '''Class to simulate from sample directory
    '''
    def __init__(self, out_port=5556, directory=None, interval=2,
                 verbose=False):
        super(Simulator, self).__init__(out_port=out_port,
                                        verbose=verbose)
        self.interval = interval
        self.directory = directory

    def run(self):
        '''run
        '''
        ex_dir = get_example_data_directory(self.directory)
        image_fpaths = glob(os.path.join(ex_dir, '*.PixelData'))
        image_fpaths.sort()
        self.logger.info('simulating %u files from %s',
                         len(image_fpaths),
                         ex_dir)
        image_fpaths = cycle(image_fpaths)

        for image_fpath in image_fpaths:
            if self.interval == 'return':
                raw_input('>> Press return for next image')
            else:
                time.sleep(self.interval)
            self.send_image(image_fpath)
            time.sleep(0.2)  # simulate image scan and reconstruction time


class Collector(object):
    '''Class to manage monitoring and loading from a directory
    '''
    def __init__(self, out_port=5556, directory=None, parent_directory=False,
                 extension='.PixelData', verbose=False):
        '''Monitor a directory
        Args:
            out_port: port to publish images to
            directory: path that will be monitored for new files
            parent_directory: if True, indicates that `directory` should be
                              monitored for the first new directory and then
                              that directory should be monitored for new files
            extension: file extension to detect
            verbose: if True, log to console
        '''
        if parent_directory:
            directory = detect_new_directory(directory)

        observer = Observer()
        handler = MonitorDirectory(out_port, extension, verbose)
        observer.schedule(handler, path=directory)

        self.observer = observer

    def run(self):
        '''Run continuously
        '''
        self.observer.start()
        try:
            while True:
                time.sleep(0.001)
        except KeyboardInterrupt:
            self.observer.stop()

        self.observer.join()


class MonitorDirectory(DataCollector, PatternMatchingEventHandler):
    '''Monitor a directory for newly created files
    '''
    def __init__(self, out_port, extension, verbose=False):
        DataCollector.__init__(self, out_port=out_port, verbose=verbose)
        PatternMatchingEventHandler.__init__(self, patterns=['*'+extension])

    def on_created(self, event):
        '''After an file creation is detected send the image
        '''
        self.send_image(event.src_path)


def detect_new_directory(directory):
    '''Monitors a directory and returns the first new directory is
       is created inside
    '''
    class MonitorNewDirectory(FileSystemEventHandler):
        '''Event handler that does the monitoring and stops itself upon
           detection of first new directory
        '''
        def __init__(self, observer):
            super(MonitorNewDirectory, self).__init__()
            self.observer = observer
            self.detected_directory = None

        def on_created(self, event):
            if event.is_directory:
                self.detected_directory = event.src_path
                self.observer.stop()

    observer = Observer()
    handler = MonitorNewDirectory(observer)
    observer.schedule(handler, path=directory)
    observer.start()
    observer.join()
    return handler.detected_directory

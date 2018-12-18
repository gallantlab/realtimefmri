import multiprocessing
import os
import signal
import threading


class TaskProxy(threading.Thread):
    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.target_process = None

    def run(self):
        p = multiprocessing.Process(target=self.target,
                                    args=self.args, kwargs=self.kwargs)
        self.target_process = p
        p.start()
        p.join()


def start_task(target, *args, **kwargs):
    t = TaskProxy(target, *args, **kwargs)
    t.daemon = True
    t.start()
    return t.target_process


def kill_process(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

import os
import os.path as op
import shutil
from uuid import uuid4
import logging
import multiprocessing
import threading
import signal
import time
import flask
from flask_caching import Cache
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import redis

from realtimefmri.control_panel import layout
from realtimefmri import collect_volumes
from realtimefmri import collect_ttl
from realtimefmri import collect
from realtimefmri import preprocess
from realtimefmri import config


external_stylesheets = [] #'https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask('app')
app = dash.Dash('app', server=server, external_stylesheets=external_stylesheets)
app.layout = layout.serve_layout
r = redis.Redis(config.REDIS_HOST)
r.flushall()

cache = Cache(app.server, config={'CACHE_TYPE': 'redis', 'CACHE_THRESHOLD': 200})


start_time = time.time()
def wait_to_start(wait_time=3):
    elapsed_time = (time.time() - start_time)
    if elapsed_time > wait_time:
        return True
    else:
        print(f"{elapsed_time} less than {wait_time}. Not starting")
        return False


class TaskProxy(threading.Thread):
    def __init__(self, target, *args):
        super().__init__()
        self.target = target
        self.args = args

    def run(self):
        p = multiprocessing.Process(target=self.target, args=self.args)
        self.target_process = p
        p.start()
        print(p.pid)
        p.join()


def start_task(target, *args):
    t = TaskProxy(target, *args)
    t.daemon = True
    t.start()
    return t.target_process


@app.callback(Output('collect-ttl-status', 'children'),
              [Input('collect-ttl-status', 'n_clicks')],
              [State('session-id', 'children')])
def collect_ttl_status(n, session_id):
    if wait_to_start():
        pid = r.get(session_id + '_collect_ttl_pid')
        if pid is None:
            label = 'o'
            process = start_task(collect_ttl.collect_ttl, 'redis')
            while not process.is_alive():
                time.sleep(0.1)
            r.set(session_id + '_collect_ttl_pid', process.pid)
        else:
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_collect_ttl_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('collect-volumes-status', 'children'),
              [Input('collect-volumes-status', 'n_clicks')],
              [State('session-id', 'children')])
def collect_volumes_status(n, session_id):
    if wait_to_start():
        pid = r.get(session_id + '_collect_volumes_pid')
        if pid is None:
            label = 'o'
            process = start_task(collect_volumes.collect_volumes)
            while not process.is_alive():
                time.sleep(0.1)
            r.set(session_id + '_collect_volumes_pid', process.pid)
        else:
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_collect_volumes_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('collect-status', 'children'),
              [Input('collect-status', 'n_clicks')],
              [State('session-id', 'children')])
def collect_status(n, session_id):
    if wait_to_start():
        pid = r.get(session_id + '_collect_pid')
        if pid is None:
            label = 'o'
            process = start_task(collect.collect)
            while not process.is_alive():
                time.sleep(0.1)
            r.set(session_id + '_collect_pid', process.pid)
        else:
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_collect_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('preprocess-status', 'children'),
              [Input('preprocess-status', 'n_clicks')],
              [State('recording-id', 'value'),
               State('preproc-config', 'value'),
               State('session-id', 'children')])
def preprocess_status(n, recording_id, preproc_config, session_id):
    if wait_to_start():
        pid = r.get(session_id + '_preprocess_pid')
        if pid is None:
            label = 'o'
            process = start_task(preprocess.preprocess, recording_id, preproc_config)
            while not process.is_alive():
                time.sleep(0.1)
            r.set(session_id + '_preprocess_pid', process.pid)
        else:
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_preprocess_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('empty-div4', 'children'),
              [Input('simulate-ttl', 'n_clicks')])
def simulate_ttl(n):
    if wait_to_start():
        print('simulating ttl at {}'.format(time.time()))
        logging.info('simulating ttl')
        r.publish('ttl', 'message')
    else:
        raise PreventUpdate()


@app.callback(Output('empty-div3', 'children'),
              [Input('simulate-volume', 'n_clicks')],
              [State('simulated-dataset', 'value'),
               State('session-id', 'children')])
def simulate_volume(n, simulated_dataset, session_id):
    if wait_to_start():
        paths = config.get_dataset_volume_paths(simulated_dataset)

        if len(paths) > 0:
            count = int(r.get(session_id + '_simulated_volume_count') or 0)
            count = count % len(paths)

            print('Simulating volume {}'.format(count))
            path = paths[count]
            shutil.copy(path, op.join(config.SCANNER_DIR, str(uuid4()) + '.dcm'))
            count += 1
            r.set(session_id + '_simulated_volume_count', count)

    raise PreventUpdate()


def serve(host='0.0.0.0', port=8051):
    app.run_server(host=host, port=port, debug=True)


if __name__ == "__main__":
    serve()

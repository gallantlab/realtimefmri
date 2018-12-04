import os
import os.path as op
import shutil
from uuid import uuid4
import logging
import multiprocessing
import threading
import signal
import time
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import redis
from realtimefmri import collect_volumes
from realtimefmri import collect_ttl
from realtimefmri import collect
from realtimefmri import preprocess
from realtimefmri import config
from realtimefmri import viewer
from realtimefmri.web_interface.app import app
from realtimefmri.utils import get_logger


logger = get_logger('control_panel', to_console=True, to_network=True)

session_id = 'admin'
layout = html.Div([html.Div(session_id, id='session-id'),  # , style={'display': 'none'}),
                   html.Div([dcc.Input(id='recording-id',
                                       placeholder='...enter recording id...',
                                       type='text', value='TEST')]),

                   # TTL status
                   html.Div([html.Button('x', id='collect-ttl-status',
                                         className='status-indicator'),
                             html.Span('Collect TTL', className='status-label'),
                             html.Button('Simulate TTL', id='simulate-ttl')]),

                   # volumes status
                   html.Div([html.Button('x', id='collect-volumes-status',
                                         className='status-indicator'),
                             html.Span('Collect volumes', className='status-label'),
                             html.Button('Simulate volume', id='simulate-volume'),
                             dcc.Dropdown(id='simulated-dataset', value='',
                                          options=[{'label': d, 'value': d}
                                                   for d in config.get_datasets()],
                                          style={'display': 'inline-block', 'width': '200px'})]),

                   # collect status
                   html.Div([html.Button('x', id='collect-status', className='status-indicator'),
                             html.Span('Collect', className='status-label')]),

                   # preprocess status
                   html.Div([html.Button('x', id='preprocess-status', 
                                         className='status-indicator'),
                             html.Span('Preprocess', className='status-label'),
                             dcc.Dropdown(id='preproc-config', value='',
                                          options=[{'label': p, 'value': p}
                                                   for p in config.get_pipelines('preproc')],
                                          style={'display': 'inline-block', 'width': '200px'})]),
                   # pycortex viewer status
                   html.Div([html.Button('x', id='viewer-status',
                                         className='status-indicator'),
                             html.Span('Viewer', className='status-label'),
                             dcc.Dropdown(id='pycortex-surface', value='',
                                          options=[{'label': d, 'value': d}
                                                   for d in config.get_surfaces()],
                                          style={'display': 'inline-block', 'width': '200px'}),
                             dcc.Dropdown(id='pycortex-transform',
                                          style={'display': 'inline-block', 'width': '200px'}),
                             dcc.Input(id='pycortex-mask', placeholder='',
                                       type='text', value='',
                                       style={'display': 'inline-block', 'width': '200px'})]),

                   html.Div(id='empty-div1', children=[]),
                   html.Div(id='empty-div2', children=[]),
                   html.Div(id='empty-div3', children=[]),
                   html.Div(id='empty-div4', children=[])],
                  style={'max-width': '600px'})

r = redis.StrictRedis(config.REDIS_HOST)
r.flushall()


class TaskProxy(threading.Thread):
    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs

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


@app.callback(Output('collect-ttl-status', 'children'),
              [Input('collect-ttl-status', 'n_clicks')],
              [State('session-id', 'children')])
def collect_ttl_status(n, session_id):
    if n is not None:
        pid = r.get(session_id + '_collect_ttl_pid')
        if pid is None:
            label = 'o'
            process = start_task(collect_ttl.collect_ttl, 'redis')
            while not process.is_alive():
                time.sleep(0.1)
            logger.info(f"Started TTL collector (pid {process.pid})")
            r.set(session_id + '_collect_ttl_pid', process.pid)
        else:
            logger.info(f"Stopping TTL collector (pid {pid})")
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
    if n is not None:
        pid = r.get(session_id + '_collect_volumes_pid')
        if pid is None:
            label = 'o'
            process = start_task(collect_volumes.collect_volumes_poll,
                                 parent_directory=config.SCANNER_DIR, extension='.dcm')
            while not process.is_alive():
                time.sleep(0.1)
            logger.info(f"Started volume collector (pid {process.pid})")
            r.set(session_id + '_collect_volumes_pid', process.pid)
        else:
            logger.info(f"Stopping volume collector (pid {pid})")
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
    if n is not None:
        pid = r.get(session_id + '_collect_pid')
        if pid is None:
            label = 'o'
            process = start_task(collect.collect)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info(f"Started collector viewer (pid {process.pid})")
            r.set(session_id + '_collect_pid', process.pid)
        else:
            logger.info(f"Stopping collector viewer (pid {pid})")
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_collect_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('preprocess-status', 'children'),
              [Input('preprocess-status', 'n_clicks')],
              [State('session-id', 'children'),
               State('recording-id', 'value'),
               State('preproc-config', 'value'),
               State('pycortex-surface', 'value'),
               State('pycortex-transform', 'value')])
def preprocess_status(n, session_id, recording_id, preproc_config, surface, transform):
    if n is not None:
        pid = r.get(session_id + '_preprocess_pid')
        if pid is None:
            label = 'o'
            process = start_task(preprocess.preprocess,
                                 recording_id, preproc_config, surface, transform)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info(f"Started preprocessor (pid {process.pid})")
            r.set(session_id + '_preprocess_pid', process.pid)
        else:
            logger.info(f"Stopping preprocessor (pid {pid})")
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_preprocess_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('viewer-status', 'children'),
              [Input('viewer-status', 'n_clicks')],
              [State('session-id', 'children'),
               State('pycortex-surface', 'value'),
               State('pycortex-transform', 'value'),
               State('pycortex-mask', 'value')])
def viewer_status(n, session_id, surface, transform, mask):
    if n is not None:
        pid = r.get(session_id + '_viewer_pid')
        if pid is None:
            label = 'o'
            process = start_task(viewer.serve, surface, transform, mask, 0, 2000)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info(f"Started pycortex viewer (pid {process.pid})")
            r.set(session_id + '_viewer_pid', process.pid)
        else:
            logger.info(f"Stopping pycortex viewer (pid {pid})")
            label = 'x'
            pid = int(pid)
            os.kill(pid, signal.SIGKILL)
            r.delete(session_id + '_viewer_pid')

        return label

    else:
        raise PreventUpdate()


@app.callback(Output('empty-div4', 'children'),
              [Input('simulate-ttl', 'n_clicks')])
def simulate_ttl(n):
    if n is not None:
        logging.info('simulating ttl at {}'.format(time.time()))
        r.publish('ttl', 'message')
    else:
        raise PreventUpdate()


@app.callback(Output('empty-div3', 'children'),
              [Input('simulate-volume', 'n_clicks')],
              [State('simulated-dataset', 'value'),
               State('session-id', 'children')])
def simulate_volume(n, simulated_dataset, session_id):
    if n is not None:
        paths = config.get_dataset_volume_paths(simulated_dataset)

        count = r.get(session_id + '_simulated_volume_count')
        if count is None:
            count = 0

            dest_directory = op.join(config.SCANNER_DIR, str(uuid4()))
            os.makedirs(dest_directory)
            time.sleep(0.5)
            logger.info(f'Simulating volume {count} {dest_directory}')
            paths = config.get_dataset_volume_paths(simulated_dataset)
            r.set(session_id + '_simulated_volume_directory', dest_directory)

        else:
            count = int(count)
            dest_directory = r.get(session_id + '_simulated_volume_directory').decode('utf-8')

        path = paths[count % len(paths)]
        dest_path = op.join(dest_directory, f"IM{count:04}.dcm")
        logger.info(f'Copying {path} to {dest_directory}')
        shutil.copy(path, dest_path)
        count += 1
        r.set(session_id + '_simulated_volume_count', count)

    raise PreventUpdate()


@app.callback(Output('pycortex-transform', 'options'),
              [Input('pycortex-surface', 'value')])
def populate_transforms(surface):
    return [{'label': d, 'value': d} for d in config.get_transforms(surface)]

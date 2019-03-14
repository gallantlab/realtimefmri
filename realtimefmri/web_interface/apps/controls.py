# -*- coding: utf-8 -*-
import logging
import os
import os.path as op
import pickle
import shutil
import time
from datetime import datetime
from uuid import uuid4

import dash
import dash_core_components as dcc
import dash_html_components as html
import redis
from dash.dependencies import Input, Output, State

from realtimefmri import collect, collect_ttl, config, preprocess, viewer
from realtimefmri.utils import get_logger
from realtimefmri.web_interface import utils
from realtimefmri.web_interface.app import app

logger = get_logger('control_panel', to_console=True, to_network=True)


def create_control_button(label, button_id):
    return html.Div([html.Button(u'▶', id=button_id, className='status-indicator'),
                     html.Span(label, className='status-label')], className='control-panel-button')


session_id = 'admin'
ttl_sources = ['redis', 'keyboard']
datasets = config.get_datasets()

layout = [
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    html.Div([html.Div(session_id, id='session-id'),
              dcc.Input(id='recording-id', placeholder='Recording id...',
                        type='text', value=datetime.strftime(datetime.now(), '%Y%m%d')),
              html.Button(u'🗑', id='flush-db')],
             id='top-bar'),

    # collect
    html.Div(className='control-panel', children=[
        create_control_button('Collect TTL', 'collect-ttl-status'),
        create_control_button('Collect', 'collect-status'),
        html.Div([html.Span('TR: ', style={'font-size': 'x-small'}),
                  html.Span('', id='tr-count', style={'font-size': 'x-small'}),
                  html.Span(' / ', style={'font-size': 'x-small'}),
                  html.Span('Trial: ', style={'font-size': 'x-small'}),
                  html.Span('', id='trial-count', style={'font-size': 'x-small'})]),
        html.Hr(), html.Span('Simulation', style={'font-size': 'x-small'}),
        dcc.Dropdown(id='ttl-source', className='control-panel-dropdown',
                     placeholder='TTL source...', value=ttl_sources[0],
                     options=[{'label': d, 'value': d} for d in ttl_sources]),
        dcc.Dropdown(id='simulated-dataset', className='control-panel-dropdown',
                     placeholder='Simulation dataset...', value=datasets[0],
                     options=[{'label': d, 'value': d} for d in datasets]),
        html.Button('TTL', id='simulate-ttl'), html.Button('DCM', id='simulate-volume'),
        create_control_button('Simulate experiment', 'simulate-experiment'),
        dcc.Input(id='simulated-tr', placeholder='Simulation TR',
                  type='number', value=2.0045)
    ]),

    # preprocess
    html.Div(className='control-panel', children=[
        create_control_button('Preprocess', 'preprocess-status'),
        html.Hr(), html.Span('Options', style={'font-size': 'x-small'}),
        dcc.Dropdown(id='preproc-config', className='control-panel-dropdown',
                     placeholder='Configuration...', value='',
                     options=[{'label': p, 'value': p} for p in config.get_pipelines('preproc')]),
        dcc.Dropdown(id='pycortex-surface', className='control-panel-dropdown',
                     placeholder='Surface...', value='',
                     options=[{'label': d, 'value': d} for d in config.get_surfaces()]),
        dcc.Dropdown(id='pycortex-transform', className='control-panel-dropdown',
                     placeholder='Transform...', value=''),
        dcc.Dropdown(id='pycortex-mask', className='control-panel-dropdown',
                     placeholder='Mask...', value='')]),

    # pycortex viewer
    html.Div(className='control-panel', children=[
        create_control_button('Viewer', 'viewer-status')]),

    html.Div(id='empty-div1', children=[]),
    html.Div(id='empty-div2', children=[]),
    html.Div(id='empty-div3', children=[])
]

r = redis.StrictRedis(config.REDIS_HOST)


@app.callback(Output('empty-div1', 'children'),
              [Input('flush-db', 'n_clicks')])
def flush_db(n):
    if n is not None:
        r.flushdb()

    raise dash.exceptions.PreventUpdate()


@app.callback(Output('tr-count', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_tr_count(n):
    if n is not None:
        count = r.get('image_number')
        if count:
            count = pickle.loads(count)

        return count

    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('trial-count', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_trial_index(n):
    if n is not None:
        trial_count = r.get('experiment:trial:current')
        if trial_count:
            trial_count = pickle.loads(trial_count)['index']

        return trial_count

    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('collect-ttl-status', 'children'),
              [Input('collect-ttl-status', 'n_clicks')],
              [State('ttl-source', 'value')])
def collect_ttl_status(n, ttl_source):
    if n is not None:
        pid = r.get(session_id + '_collect_ttl_pid')
        if pid is None:
            label = u'■'
            process = utils.start_task(collect_ttl.collect_ttl, ttl_source)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info("Started TTL collector (pid %d)", process.pid)
            r.set(session_id + '_collect_ttl_pid', process.pid)
        else:
            logger.info("Stopping TTL collector (pid %s)", pid)
            label = u'▶'
            utils.kill_process(int(pid))
            r.delete(session_id + '_collect_ttl_pid')

        return label

    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('collect-status', 'children'),
              [Input('collect-status', 'n_clicks')])
def collect_status(n):
    if n is not None:
        pid = r.get(session_id + '_collect_pid')
        if pid is None:
            label = u'■'
            process = utils.start_task(collect.collect)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info("Started collector (pid %d)", process.pid)
            r.set(session_id + '_collect_pid', process.pid)
        else:
            logger.info("Stopping collector (pid %s)", pid)
            label = u'▶'
            utils.kill_process(int(pid))
            r.delete(session_id + '_collect_pid')

        return label

    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('preprocess-status', 'children'),
              [Input('preprocess-status', 'n_clicks')],
              [State('recording-id', 'value'),
               State('preproc-config', 'value'),
               State('pycortex-surface', 'value'),
               State('pycortex-transform', 'value'),
               State('pycortex-mask', 'value')])
def preprocess_status(n, recording_id, preproc_config, surface, transform, mask):
    if n is not None:
        pid = r.get(session_id + '_preprocess_pid')
        if pid is None:
            label = u'■'
            global_parameters = {'surface': surface, 'transform': transform, 'mask_type': mask}
            process = utils.start_task(preprocess.preprocess,
                                       recording_id, preproc_config, **global_parameters)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info("Started preprocessor (pid %d)", process.pid)
            r.set(session_id + '_preprocess_pid', process.pid)
        else:
            logger.info("Stopping preprocessor (pid %s)", pid)
            label = u'▶'
            utils.kill_process(int(pid))
            r.delete(session_id + '_preprocess_pid')

        return label

    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('viewer-status', 'children'),
              [Input('viewer-status', 'n_clicks')],
              [State('pycortex-surface', 'value'),
               State('pycortex-transform', 'value'),
               State('pycortex-mask', 'value')])
def viewer_status(n, surface, transform, mask):
    if n is not None:
        pid = r.get(session_id + '_viewer_pid')
        if pid is None:
            label = u'■'
            process = utils.start_task(viewer.serve, surface, transform, mask)
            while not process.is_alive():
                time.sleep(0.1)
            logger.info("Started pycortex viewer (pid %d)", process.pid)
            r.set(session_id + '_viewer_pid', process.pid)
        else:
            logger.info("Stopping pycortex viewer (pid %s)", pid)
            label = u'▶'
            utils.kill_process(int(pid))
            r.delete(session_id + '_viewer_pid')

        return label

    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('empty-div2', 'children'),
              [Input('simulate-ttl', 'n_clicks')])
def simulate_ttl(n):
    if n is not None:
        logging.info('Simulating ttl at %s', str(time.time()))
        r.publish('ttl', 'message')
    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('empty-div3', 'children'),
              [Input('simulate-volume', 'n_clicks')],
              [State('simulated-dataset', 'value')])
def simulate_volume(n, simulated_dataset):
    if n is not None:
        paths = config.get_dataset_volume_paths(simulated_dataset)

        count = r.get(session_id + '_simulated_volume_count')
        if count is None:
            count = 0
            dest_directory = str(uuid4())
            os.makedirs(op.join(config.SCANNER_DIR, dest_directory))
            time.sleep(0.5)
            logger.info('Simulating volume %d %s', count, dest_directory)
            r.set(session_id + '_simulated_volume_directory', dest_directory)

        else:
            count = int(count)
            dest_directory = r.get(session_id + '_simulated_volume_directory').decode('utf-8')

        path = paths[count % len(paths)]
        dest_path = op.join(config.SCANNER_DIR, dest_directory, f"IM{count:04}.dcm")
        logger.info('Copying %s to %s', path, dest_path)
        shutil.copy(path, dest_path)
        count += 1
        r.set(session_id + '_simulated_volume_count', count)

    raise dash.exceptions.PreventUpdate()


@app.callback(Output('simulate-experiment', 'children'),
              [Input('simulate-experiment', 'n_clicks')],
              [State('simulated-dataset', 'value'),
               State('simulated-tr', 'value')])
def simulate_experiment(n, simulated_dataset, TR):
    if n is not None:
        pid = r.get(session_id + '_simulate_experiment_pid')
        if pid is None:
            label = u'■'
            process = utils.start_task(simulate_experiment_process,
                                       simulated_dataset, TR)
            logger.info("Started simulation of experiment (pid %d)",
                        process.pid)
            r.set(session_id + '_simulate_experiment_pid', process.pid)
        else:
            logger.info("Stopping simulation of experiment (pid %s)", pid)
            label = u'▶'
            utils.kill_process(int(pid))
            r.delete(session_id + '_simulate_experiment_pid')
        return label
    else:
        raise dash.exceptions.PreventUpdate()


def simulate_experiment_process(simulated_dataset, TR):
    r = redis.StrictRedis(config.REDIS_HOST)
    paths = config.get_dataset_volume_paths(simulated_dataset)
    count = 0
    dest_directory = str(uuid4())
    os.makedirs(op.join(config.SCANNER_DIR, dest_directory))
    time.sleep(0.5)
    for path in paths:
        logger.info('Simulating volume %d %s', count, dest_directory)
        dest_path = op.join(config.SCANNER_DIR, dest_directory, f"IM{count:04}.dcm")
        logger.info('Copying %s to %s', path, dest_path)
        logging.info('Simulating ttl at %s', str(time.time()))
        # send a TTL
        r.publish('ttl', 'message')
        # copy volume
        shutil.copy(path, dest_path)
        # wait for TR
        time.sleep(TR)
        count += 1
        r.set(session_id + '_simulated_volume_count', count)


@app.callback(Output('pycortex-transform', 'options'),
              [Input('pycortex-surface', 'value')])
def populate_transforms(surface):
    return [{'label': d, 'value': d} for d in config.get_available_transforms(surface)]


@app.callback(Output('pycortex-mask', 'options'),
              [Input('pycortex-transform', 'value')],
              [State('pycortex-surface', 'value')])
def populate_masks(transform, surface):
    return [{'label': d, 'value': d} for d in config.get_available_masks(surface, transform)]

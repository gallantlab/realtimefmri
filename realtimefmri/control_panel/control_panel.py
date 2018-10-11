import os.path as op
import shutil
from uuid import uuid4
import logging
import flask
from flask_caching import Cache
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import redis
from realtimefmri.control_panel import layout
from realtimefmri import config
from realtimefmri.collect_volumes import simulate_volumes

server = flask.Flask('app')
app = dash.Dash('app', server=server)
app.layout = layout.serve_layout
r = redis.Redis(config.REDIS_HOST)
cache = Cache(app.server, config={'CACHE_TYPE': 'redis', 'CACHE_THRESHOLD': 200})


@app.callback(Output('empty-div2', 'children'),
              [Input('start-collection', 'n_clicks')],
              [State('recording-id', 'value')])
def start_collection(n, recording_id):
    print("start_collection", n)
    cmd = ['realtimefmri', 'collect', recording_id]
    print(' '.join(cmd))
    raise PreventUpdate()


@app.callback(Output('empty-div1', 'children'),
              [Input('start-recording', 'n_clicks')],
              [State('recording-id', 'value'),
               State('preproc-config', 'value'),
               State('stim-config', 'value')])
def start_recording(n, recording_id, preproc_config, stim_config):
    print(recording_id, preproc_config, stim_config)
    cmd = ['realtimefmri', 'preprocess', recording_id, preproc_config, stim_config]
    # pid = Popen(cmd)
    print(' '.join(cmd))
    raise PreventUpdate()


@app.callback(Output('empty-div4', 'children'),
              [Input('simulate-tr', 'n_clicks')])
def simulate_tr(n):
    print('simulating ttl')
    logging.info('simulating ttl')
    r.publish('ttl', 'message')


@app.callback(Output('empty-div3', 'children'),
              [Input('simulate-volume', 'n_clicks')],
              [State('simulated-dataset', 'value'),
               State('session-id', 'children')])
def simulate_volume(n, simulated_dataset, session_id):
    paths = config.get_dataset_volume_paths(simulated_dataset)

    count = r.get(session_id + '_simulated_volume_count')
    if count is None:
        count = 0
    else:
        count = int(count)

    count = count % len(paths)

    print('Simulating volume {}'.format(count))
    path = paths[count]
    shutil.copy(path, op.join(config.SCANNER_DIR, str(uuid4()) + '.dcm'))
    count += 1
    r.set(session_id + '_simulated_volume_count', count)
    raise PreventUpdate()


def serve(host='0.0.0.0', port=8051):
    app.run_server(host=host, port=port, debug=True)

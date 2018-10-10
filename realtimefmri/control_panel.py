import os
import os.path as op
import logging
from uuid import uuid4
from subprocess import Popen
import flask
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import redis
from realtimefmri.config import (SCANNER_DIR, REDIS_HOST,
                                 get_pipelines, get_datasets)


server = flask.Flask('app')
app = dash.Dash('app', server=server)

app.layout = html.Div([html.Div([dcc.Input(id='recording-id',
                                           placeholder='...enter recording id...',
                                           type='text', value=''),
                                 dcc.Dropdown(id='preproc-config',
                                              options=[{'label': p, 'value': p}
                                                       for p in get_pipelines('preproc')],
                                              value=''),
                                 dcc.Dropdown(id='stim-config',
                                              options=[{'label': p,  'value': p}
                                                       for p in get_pipelines('stim')],
                                              value=''),
                                 html.Button('Start recording', id='start-recording')]),
                       html.Div([dcc.Dropdown(id='simulated-dataset',
                                              options=[{'label': d, 'value': d}
                                                       for d in get_datasets()],
                                              value=''),
                                 html.Button('Start collection', id='start-collection'),
                                 html.Button('Start simulation', id='start-simulation'),
                                 html.Button('Simulate TR', id='simulate-tr'),
                                 html.Button('Simulate volume', id='simulate-volume')]),
                       html.Div(id='empty-div1', children=[]),
                       html.Div(id='empty-div2', children=[]),
                       html.Div(id='empty-div3', children=[]),
                       html.Div(id='empty-div4', children=[]),

                       ])

r = redis.Redis(REDIS_HOST)
print(REDIS_HOST)

@app.callback(Output('empty-div2', 'children'),
              [Input('start-collection', 'n_clicks')],
              [State('recording-id', 'value')])
def start_collection(n, recording_id):
    print("start_collection", n)
    cmd = ['realtimefmri', 'collect', recording_id]
    # pid = Popen(cmd)
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
              [State('simulated-dataset', 'value')])
def simulate_volume(n, simulated_dataset):
    print(simulated_dataset)
    raise PreventUpdate()


def main(host='0.0.0.0', port=8051):
    app.run_server(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()

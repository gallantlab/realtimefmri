import os
import os.path as op
from uuid import uuid4
from subprocess import Popen
import flask
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
from realtimefmri.interface import collect, preprocess
from realtimefmri.config import SCANNER_DIR, get_pipelines, get_datasets
from realtimefmri.utils import simulate_keystroke


server = flask.Flask('app')
app = dash.Dash('app', server=server)

app.layout = html.Div([html.Div([dcc.Input(id='recording-id',
                                           placeholder='...enter recording id...',
                                           type='text', value=''),
                                 dcc.Dropdown(id='preproc-config',
                                              options=get_pipelines('preproc'),
                                              value=''),
                                 dcc.Dropdown(id='stim-config',
                                              options=get_pipelines('stim'),
                                              value=''),
                                 html.Button('Start recording', id='start-recording')]),
                       html.Div([dcc.Input(id='scanner-directory', type='text',
                                           value=SCANNER_DIR, style={'width': '50%'}),
                                 dcc.Input(id='volume-directory', type='text',
                                           value='', style={'width': '50%'}),
                                 dcc.Dropdown(id='simulated-dataset',
                                              options=get_datasets(),
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


@app.callback(Output('empty-div2', 'children'),
              [Input('start-collection', 'n_clicks')],
              [State('recording-id', 'value'),
               State('scanner-directory', 'value')])
def start_collection(n, recording_id, scanner_directory):
    print("start_collection", n)
    cmd = ['realtimefmri', 'collect', recording_id, '--parent_directory', scanner_directory]
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


@app.callback(Output('volume-directory', 'value'),
              [Input('start-simulation', 'n_clicks')])
def start_simulation(n):
    dest_directory = op.join(SCANNER_DIR, str(uuid4()))
    os.makedirs(dest_directory)

    return dest_directory


@app.callback(Output('empty-div4', 'children'),
              [Input('simulate-tr', 'n_clicks')])
def simulate_tr(n):
    print('simulating TR')
    simulate_keystroke()
    raise PreventUpdate()


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

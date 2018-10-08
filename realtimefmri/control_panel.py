import os.path as op
from glob import glob
from subprocess import Popen
import flask
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
from realtimefmri.interface import collect, preprocess
from realtimefmri.config import PIPELINE_DIR, SCANNER_DIR


def get_pipeline_options(pipeline_type):
    paths = sorted(glob(op.join(PIPELINE_DIR, pipeline_type + '-*.yaml')))
    options = []
    for path in paths:
        label = op.splitext(op.basename(path))[0]
        options.append({'label': label, 'value': label})

    return options


server = flask.Flask('app')
app = dash.Dash('app', server=server)

app.layout = html.Div([html.Div([dcc.Input(id='recording-id',
                                           placeholder='...enter recording id...',
                                           type='text', value=''),
                                 dcc.Dropdown(id='preproc-config',
                                              options=get_pipeline_options('preproc'),
                                              value=''),
                                 dcc.Dropdown(id='stim-config',
                                              options=get_pipeline_options('stim'),
                                              value=''),
                                 html.Button('Start recording', id='start-recording')]),
                       html.Div([dcc.Input(id='scanner-directory', type='text',
                                           value=SCANNER_DIR),
                                 html.Button('Start collection', id='start-collection'),
                                 dcc.Checklist(id='simulate',
                                               options=[{'label': 'simulate',
                                                         'value': 'simulate'}],
                                               values=[])]),
                       html.Div(id='empty-div1', children=[]),
                       html.Div(id='empty-div2', children=[])])


@app.callback(Output('empty-div1', 'children'),
              [Input('start-recording', 'n_clicks')],
              [State('recording-id', 'value'),
               State('preproc-config', 'value'),
               State('stim-config', 'value')])
def start_recording(n, recording_id, preproc_config, stim_config):
    print(recording_id, preproc_config, stim_config)
    pid = Popen(['realtimefmri', 'preprocess', recording_id, preproc_config, stim_config])
    raise PreventUpdate()


@app.callback(Output('empty-div2', 'children'),
              [Input('start-collection', 'n_clicks')],
              [State('recording-id', 'value'),
               State('scanner-directory', 'value'),
               State('simulate', 'values')])
def start_collection(n, recording_id, scanner_directory, simulate):
    simulate = len(simulate) == 1
    cmd = ['realtimefmri', 'collect', recording_id]
    if simulate:
        cmd.append('--simulate')
    cmd += ['--parent_directory', scanner_directory]
    pid = Popen(cmd)
    raise PreventUpdate()


def main(host='0.0.0.0', port=8051):
    app.run_server(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()

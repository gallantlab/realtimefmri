import dash_core_components as dcc
import dash_html_components as html
from realtimefmri import config
from uuid import uuid4


def serve_layout():
    session_id = str(uuid4())
    return html.Div([html.Div(session_id, id='session-id', style={'display': 'none'}),
                     html.Div([dcc.Input(id='recording-id',
                                         placeholder='...enter recording id...',
                                         type='text', value=''),
                               dcc.Dropdown(id='preproc-config',
                                            options=[{'label': p, 'value': p}
                                                     for p in config.get_pipelines('preproc')],
                                            value=''),
                               dcc.Dropdown(id='stim-config',
                                            options=[{'label': p, 'value': p}
                                                     for p in config.get_pipelines('stim')],
                                            value=''),
                               html.Button('Start recording', id='start-recording')]),
                     html.Div([dcc.Dropdown(id='simulated-dataset',
                                            options=[{'label': d, 'value': d}
                                                     for d in config.get_datasets()],
                                            value=''),
                               html.Button('Start collection', id='start-collection'),
                               html.Button('Start simulation', id='start-simulation'),
                               html.Button('Simulate TR', id='simulate-tr'),
                               html.Button('Simulate volume', id='simulate-volume')]),
                     html.Div(id='empty-div1', children=[]),
                     html.Div(id='empty-div2', children=[]),
                     html.Div(id='empty-div3', children=[]),
                     html.Div(id='empty-div4', children=[])])

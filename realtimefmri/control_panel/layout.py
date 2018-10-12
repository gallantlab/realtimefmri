import dash_core_components as dcc
import dash_html_components as html
from realtimefmri import config
from uuid import uuid4


def serve_layout():
    # session_id = str(uuid4())
    session_id = 'admin'
    return html.Div([html.Div(session_id, id='session-id'), # , style={'display': 'none'}),
                     html.Div([dcc.Input(id='recording-id',
                                         placeholder='...enter recording id...',
                                         type='text', value='TEST')]),

                     # TTL status
                     html.Div([html.Button('x', id='collect-ttl-status',
                                           className='status-indicator'),
                               html.Span('Collect TTL', className='collect-label'),
                               html.Button('Simulate TTL', id='simulate-ttl')]),

                     # volumes status
                     html.Div([html.Button('x', id='collect-volumes-status',
                                           className='status-indicator'),
                               html.Span('Collect volumes', className='collect-label'),
                               html.Button('Simulate volume', id='simulate-volume'),
                               dcc.Dropdown(id='simulated-dataset',
                                            options=[{'label': d, 'value': d}
                                                     for d in config.get_datasets()],
                                            value='', style={'display': 'inline-block', 'width': '200px'})]),

                     # collect status
                     html.Div([html.Button('x', id='collect-status', className='status-indicator'),
                               html.Span('Collect', className='collect-label')]),

                     # preprocess status
                     html.Div([html.Button('x', id='preprocess-status', 
                                           className='status-indicator'),
                               html.Span('Preprocess', className='collect-label'),
                               dcc.Dropdown(id='preproc-config', value='',
                                            options=[{'label': p, 'value': p}
                                                     for p in config.get_pipelines('preproc')],
                                            style={'display': 'inline-block', 'width': '200px'})]),

                     html.Div(id='empty-div1', children=[]),
                     html.Div(id='empty-div2', children=[]),
                     html.Div(id='empty-div3', children=[]),
                     html.Div(id='empty-div4', children=[])],
                    style={'max-width': '600px'})

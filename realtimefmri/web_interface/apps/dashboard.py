import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from realtimefmri.web_interface.app import app

layout = html.Div([html.H2("rtFMRI dashboard"),
                   html.Div([  # dcc.Dropdown(id='data-list', value='', multi=True)
                   		     dcc.Input(id='data-list', value='volume')],
                            id='data-list-div'),
                   html.Div([dcc.Graph(id='graphs',
                                       style={'display': 'inline-block', 'height': '90%'})],
                            id='graph-div'),
                   dcc.Interval(id='interval-component', interval=500, n_intervals=0)],
                  style={'display': 'inline-block', 'width': '100vh', 'height': '100vh'},
                  className="container")

import pickle
from collections import defaultdict
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import redis
from realtimefmri import config
from realtimefmri import preprocess
from realtimefmri.web_interface.app import app
from realtimefmri import pipeline_utils
from realtimefmri.utils import get_logger

session_id = 'admin'
logger = get_logger('pipeline', to_console=True, to_network=True)
r = redis.StrictRedis(config.REDIS_HOST)


def create_interface():
    class_name_key = list(r.scan_iter('pipeline:*:0:class_name'))[0]
    pipeline_key = class_name_key.rsplit(b':', maxsplit=2)[0]
    interface = preprocess.Pipeline.create_interface(pipeline_key)
    return interface

layout = html.Div([html.Button('x', id='refresh-interfaces'),
                   html.Div(id='pipeline-interfaces')])


@app.callback(Output('pipeline-interfaces', 'children'),
              [Input('refresh-interfaces', 'n_clicks')])
def refresh_interfaces(n):
    if n is not None:
        return create_interface()

    else:
        PreventUpdate()

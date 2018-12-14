import pickle
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import redis
from dash.dependencies import Input, Output, State

from realtimefmri import config, pipeline_utils, preprocess
from realtimefmri.utils import get_logger
from realtimefmri.web_interface.app import app

session_id = 'admin'
logger = get_logger('pipeline', to_console=True, to_network=True)
r = redis.StrictRedis(config.REDIS_HOST)


def create_interface():
    logger.debug('Create interface')
    class_name_key = list(r.scan_iter('pipeline:*:0:class_name'))[0]
    pipeline_key = class_name_key.rsplit(b':', maxsplit=2)[0]
    interface = preprocess.Pipeline.create_interface(pipeline_key)
    logger.debug('%s', str(interface))
    return interface


layout = html.Div([html.Button('Refresh interface', id='refresh-interface'),
                   html.Div(id='pipeline-interface')])


@app.callback(Output('pipeline-interface', 'children'),
              [Input('refresh-interface', 'n_clicks')])
def refresh_interface(n):
    if n is not None:
        return create_interface()

    else:
        dash.exceptions.PreventUpdate()

import pickle
from collections import defaultdict
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import redis
from realtimefmri import config
from realtimefmri import preprocess
from realtimefmri.utils import get_logger

session_id = 'admin'
logger = get_logger('control_panel', to_console=True, to_network=True)
r = redis.StrictRedis(config.REDIS_HOST)


def create_interfaces():
    contents = []
    for key in r.scan_iter(b'pipeline:*:name'):
        step_id = key.split(b':')[1].decode('utf-8')
        step_name = pickle.loads(r.get(key))
        interface = preprocess.get_interface(step_name, step_id)
        contents.append(interface)

    return html.Div(contents, id='pipeline-interfaces')

layout = html.Div(create_interfaces(),
                  style={'max-width': '600px'})

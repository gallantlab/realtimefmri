# -*- coding: utf-8 -*-
import pickle
import time
from collections import defaultdict

import dash_core_components as dcc
import dash_html_components as html
import redis
from dash.dependencies import Input, Output

from realtimefmri import config
from realtimefmri.utils import get_logger
from realtimefmri.web_interface.app import app

logger = get_logger(__name__, to_console=True, to_network=False)
graphs = defaultdict(list)
r = redis.StrictRedis(config.REDIS_HOST)

layout = [html.Video(id='video', autoPlay=True),
          dcc.Interval(id='interval-component', interval=500, n_intervals=0)]


@app.callback(Output('video', 'src'),
              [Input('interval-component', 'n_intervals')])
def update_video(n):
    src = r.get('video:src')
    if src:
        src = src.decode('utf-8')
    else:
        src = ''
    return src

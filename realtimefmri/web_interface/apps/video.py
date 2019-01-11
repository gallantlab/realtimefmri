# -*- coding: utf-8 -*-
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import redis
from dash.dependencies import Input, Output, State

from realtimefmri import config
from realtimefmri.utils import get_logger
from realtimefmri.web_interface.app import app

logger = get_logger('dashboard', to_console=True, to_network=False)
graphs = defaultdict(list)
r = redis.StrictRedis(config.REDIS_HOST)

layout = [html.Video(src='static/videos/TheScientist.mov')]

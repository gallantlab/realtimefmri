import numbers
import pickle
import warnings
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import PIL
import plotly
import plotly.graph_objs as go
import redis
from dash.dependencies import Input, Output

from realtimefmri import config
from realtimefmri.utils import get_logger
from realtimefmri.web_interface.app import app

logger = get_logger('dashboard', to_console=True, to_network=False)
graphs = defaultdict(list)
r = redis.StrictRedis(config.REDIS_HOST)


def get_data_options():
    class_name_key = list(r.scan_iter('pipeline:*:0:class_name'))[0]
    pipeline_key = class_name_key.rsplit(b':', maxsplit=2)[0]

    data_options = []
    for key in r.scan_iter(pipeline_key + b':*:class_name'):
        class_name = pickle.loads(r.get(key))
        if class_name == 'realtimefmri.stimulate.SendToDashboard':
            data_name_key = key.replace(b':class_name', b':name')
            data_name = pickle.loads(r.get(data_name_key))
            data_options.append({'label': data_name,
                                 'value': 'dashboard:data:' + data_name})

    return data_options


def remove_prefix(text, prefix):
    """Remove a prefix from text
    """
    if text.startswith(prefix):
        return text[len(prefix):]

    return text


def make_volume_slices(volume, x=None, y=None, z=None):
    logger.info('volume_slices vol shape %s', str(volume.shape))
    if x is None:
        x = volume.shape[0] // 2
    if y is None:
        y = volume.shape[1] // 2
    if z is None:
        z = volume.shape[2] // 2
    return [go.Heatmap(z=volume[x, :, :], colorscale='Greys'),
            go.Heatmap(z=volume[:, y, :], colorscale='Greys'),
            go.Heatmap(z=volume[:, :, z], colorscale='Greys')]


def generate_update_graph():
    def update_graph(n, selected_values):

        if len(selected_values) == 0:
            raise dash.exceptions.PreventUpdate()

        titles = [remove_prefix(k, 'dashboard:data:') for k in selected_values]
        fig = plotly.tools.make_subplots(rows=len(selected_values), cols=1,
                                         subplot_titles=titles, print_grid=False)

        new_layouts = []
        images = []

        for i, key in enumerate(selected_values):
            axis_number = i + 1
            dat = r.get(key)
            if dat:
                data = pickle.loads(dat)
                plot_type = r.get(key + ':type')

                if plot_type == b'bar':
                    if isinstance(data, numbers.Number):
                        data = [data]

                    trace = go.Bar(y=data)
                    fig.append_trace(trace, axis_number, 1)

                elif plot_type == b'timeseries':
                    update = r.get(key + ':update')
                    if update == b'true':
                        graphs[key].append(data)
                        r.set(key + ':update', b'false')

                    data = np.array(graphs[key])
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)

                    for trace_index in range(data.shape[1]):
                        trace = go.Scatter(y=data[:, trace_index])
                        fig.append_trace(trace, axis_number, 1)

                elif plot_type == b'image':
                    if data.dtype != np.dtype('uint8'):
                        data = ((data - np.nanmin(data)) / np.nanmax(data))
                        data[np.isnan(data)] = 0.5
                        data *= 255
                        data = data.astype('uint8')

                    if data.shape[0] < data.shape[1]:
                        data = data.T

                    img = PIL.Image.fromarray(data)

                    fig.append_trace(go.Scatter(), axis_number, 1)
                    image = {'source': img,
                             'xref': f"x{axis_number}", 'yref': f"y{axis_number}",
                             'x': 0, 'y': 1,
                             'sizex': 1, 'sizey': 1,
                             'sizing': "stretch",
                             'opacity': 1,
                             'layer': "above"}
                    images.append(image)

                    new_layout = {f'xaxis{axis_number}': {'range': [0, 1]},
                                  f'yaxis{axis_number}': {'range': [0, 1]}}
                    new_layouts.append(new_layout)
                    titles.append(remove_prefix(key, 'dashboard:data:'))

                else:
                    warnings.warn('{} plot not implemented. Omitting this plot.'.format(plot_type))
            else:
                raise dash.exceptions.PreventUpdate()

        fig.layout.update({'autosize': True})

        if len(images) > 0:
            fig.layout.update({'images': images})

        for new_layout in new_layouts:
            fig.layout.update(new_layout)

        return fig

    return update_graph


def generate_refresh_selector():
    def refresh_selector(n):
        if n is not None:
            return get_data_options()
        else:
            return dash.exceptions.PreventUpdate()

    return refresh_selector


n_graphs = 4

selector_list = []
graph_list = []
for graph_index in range(n_graphs):
    selector = dcc.Dropdown(id=f'data-selector{graph_index}',
                            value=[], multi=True, options=[])
    selector_list.append(selector)

    graph = dcc.Graph(id=f'graph{graph_index}')
    graph_list.append(graph)


layout = [html.Button('Refresh data list', id='refresh-selector-button'),
          html.Div(selector_list, id='data-selector-div'),
          html.Div(graph_list, id='graph-div'),
          dcc.Interval(id='interval-component', interval=1000, n_intervals=0)]


for graph_index in range(n_graphs):
    app.callback(Output(f'data-selector{graph_index}', 'options'),
                 [Input('refresh-selector-button', 'n_clicks')])(generate_refresh_selector())

    app.callback(Output(f'graph{graph_index}', 'figure'),
                 [Input('interval-component', 'n_intervals'),
                  Input(f'data-selector{graph_index}', 'value')])(generate_update_graph())

import pickle
import warnings
from collections import defaultdict
import numbers
import numpy as np
import PIL
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import redis
from realtimefmri import config
from realtimefmri.web_interface.app import app
from realtimefmri.utils import get_logger


logger = get_logger('dashboard', to_console=True, to_network=False)


graphs = defaultdict(list)


def remove_prefix(text, prefix):
    """Remove a prefix from text
    """
    if text.startswith(prefix):
        return text[len(prefix):]

    return text


def make_volume_slices(volume, x=None, y=None, z=None):
    logger.info(f'volume_slices vol shape {volume.shape}')
    if x is None:
        x = volume.shape[0] // 2
    if y is None:
        y = volume.shape[1] // 2
    if z is None:
        z = volume.shape[2] // 2
    return [go.Heatmap(z=volume[x, :, :], colorscale='Greys'),
            go.Heatmap(z=volume[:, y, :], colorscale='Greys'),
            go.Heatmap(z=volume[:, :, z], colorscale='Greys')]


layout = html.Div([html.Div([dcc.Dropdown(id='data-list', value='', multi=True)],
                            id='data-list-div'),
                   html.Div([dcc.Graph(id='graphs')],
                            id='graph-div'),
                   dcc.Interval(id='interval-component', interval=1000, n_intervals=0)],
                  id="container")

r = redis.StrictRedis(config.REDIS_HOST)


@app.callback(Output('data-list', 'options'),
              [Input('interval-component', 'n_intervals')])
def update_data_list(n):
    data_list = []
    for key in r.scan_iter(b'dashboard:data:*'):
        key = key.decode('utf-8')
        if len(key.split(':')) == 3:
            label = remove_prefix(key, 'dashboard:data:')
            data_list.append({'label': label,
                              'value': key})

    return data_list


@app.callback(Output('graphs', 'figure'),
              [Input('interval-component', 'n_intervals'),
               Input('data-list', 'value')])
def update_selected_graphs(n, selected_values):

    if len(selected_values) == 0:
        return go.Scatter()

    titles = [remove_prefix(k, 'dashboard:data:') for k in selected_values]
    fig = plotly.tools.make_subplots(rows=len(selected_values), cols=1,
                                     subplot_titles=titles, print_grid=False)

    new_layouts = []
    images = []

    for i, key in enumerate(selected_values):
        axis_number = i + 1
        dat = r.get(key)
        if dat is not None:
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

    fig.layout.update({'autosize': True})

    if len(images) > 0:
        fig.layout.update({'images': images})

    for new_layout in new_layouts:
        fig.layout.update(new_layout)

    return fig

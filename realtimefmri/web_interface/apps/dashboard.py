import pickle
import warnings
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
                   dcc.Interval(id='interval-component', interval=500, n_intervals=0)],
                  id="container")

r = redis.StrictRedis(config.REDIS_HOST)


@app.callback(Output('data-list', 'options'),
              [Input('interval-component', 'n_intervals')])
def update_data_list(n):
    data_list = []
    for key in r.scan_iter(b'dashboard:*'):
        key = key.decode('utf-8')
        if not key.endswith(':type'):
            label = remove_prefix(key, 'dashboard:')
            data_list.append({'label': label,
                              'value': key})

    return data_list


@app.callback(Output('graphs', 'figure'),
              [Input('interval-component', 'n_intervals'),
               Input('data-list', 'value')])
def update_selected_graphs(n, selected_values):
    fig_specs = []
    traces = []
    new_layouts = []
    images = []
    axis_number = 1

    for i, value in enumerate(selected_values):
        val = r.get(value)
        if val is not None:
            data = pickle.loads(val)
            plot_type = r.get(value + ':type')
            if plot_type == b'scatter':
                traces.append(go.Scatter(y=data))
                fig_specs.append([{'colspan': 3}, None, None])
                axis_number += 1

            elif plot_type == b'bar':
                if isinstance(data, numbers.Number):
                    data = [data]
                traces.append(go.Bar(y=data))
                fig_specs.append([{'colspan': 3}, None, None])
                axis_number += 1

            elif plot_type == b'timeseries':
                traces.append(go.Scatter(y=[1, 2, 3]))
                fig_specs.append([{'colspan': 3}, None, None])
                axis_number += 1

            elif plot_type == b'image':
                logger.info(f"min {data.min()}, max {data.max()}")
                if data.dtype != np.dtype('uint8'):
                    data = ((data - np.nanmin(data)) / np.nanmax(data))
                    data[np.isnan(data)] = 0.5
                    data *= 255
                    data = data.astype('uint8')
                logger.info(f"min {data.min()}, max {data.max()}")

                if data.shape[0] < data.shape[1]:
                    data = data.T

                img = PIL.Image.fromarray(data)
                traces.append(go.Scatter())
                fig_specs.append([{'colspan': 3}, None, None])
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
                axis_number += 1
                new_layouts.append(new_layout)

            elif plot_type == b'volume_slices':
                traces.append(make_volume_slices(data))
                fig_specs.append([{}, {}, {}])
                axis_number += 3

            else:
                warnings.warn('{} plot not implemented. Omitting this plot.'.format(plot_type))

    if len(traces) == 0:
        return go.Scatter()

    fig = plotly.tools.make_subplots(rows=len(traces), cols=3, specs=fig_specs,
                                     print_grid=False)

    for row_index, trace in enumerate(traces):
        if isinstance(trace, list):
            for column_index, tr in enumerate(trace):
                fig.append_trace(tr, row_index + 1, column_index + 1)
        else:
            fig.append_trace(trace, row_index + 1, 1)

    fig.layout.update({'autosize': True})

    if len(images) > 0:
        fig.layout.update({'images': images})

    for new_layout in new_layouts:
        fig.layout.update(new_layout)

    return fig

# -*- coding: utf-8 -*-
import numbers
import pickle
import warnings
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import PIL
import plotly.graph_objs as go
import redis
from dash.dependencies import Input, Output, State

from realtimefmri import config
from realtimefmri.utils import get_logger
from realtimefmri.web_interface.app import app

logger = get_logger('realtimefmri.dashboard', to_console=True, to_network=False, to_file=True)
graphs = defaultdict(list)
r = redis.StrictRedis(config.REDIS_HOST)


def get_data_options():
    class_name_key = list(r.scan_iter('pipeline:*:0:class_name'))[0]
    pipeline_key = class_name_key.rsplit(b':', maxsplit=2)[0]

    data_options = []
    for key in r.scan_iter(pipeline_key + b':*:class_name'):
        class_name = pickle.loads(r.get(key))
        if class_name == 'realtimefmri.preprocess.SendToDashboard':
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
    """Return a callback function for updating a graph
    """
    def update_graph(n, selected_values):
        """Callback to update a graph given a list of data names

        Parameters
        ----------
        n : int
        selected_values: list of str
        """

        if len(selected_values) == 0:
            raise dash.exceptions.PreventUpdate()

        traces = []
        images = []
        layout_updates = []
        titles = []

        for key in selected_values:
            dat = r.get(key)
            title = remove_prefix(key, 'dashboard:data:')
            titles.append(title)
            if dat:
                logger.debug('dashboard %s', len(dat))
                data = pickle.loads(dat)
                plot_type = r.get(key + ':type')

                if plot_type == b'bar':
                    if isinstance(data, numbers.Number):
                        data = [data]

                    trace = go.Bar(y=data)
                    traces.append(trace)

                elif plot_type == b'timeseries':
                    update = r.get(key + ':update')
                    if update == b'true':
                        logger.debug('Appending to timeseries %s', str(data))
                        graphs[key].append(data)
                        r.set(key + ':update', b'false')

                    data = np.array(graphs[key])
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)

                    logger.debug('Timeseries shape %s', str(data.shape))
                    for trace_index in range(data.shape[1]):
                        trace = go.Scatter(y=data[:, trace_index], name=trace_index)
                        traces.append(trace)

                elif plot_type == b'array_image':
                    data = data[0]
                    if data.dtype != np.dtype('uint8'):
                        data = ((data - np.nanmin(data)) / np.nanmax(data))
                        data[np.isnan(data)] = 0.5
                        data *= 255
                        data = data.astype('uint8')

                    if data.shape[0] < data.shape[1]:
                        data = data.T

                    img = PIL.Image.fromarray(data)

                    traces.append(go.Scatter())
                    image = {'source': img,
                             'xref': 'x', 'yref': 'y',
                             'x': 0, 'y': 1,
                             'sizex': 1, 'sizey': 1,
                             'sizing': "stretch",
                             'opacity': 1,
                             'layer': "above"}
                    images.append(image)

                    layout = {f'xaxis': {'range': [0, 1]},
                              f'yaxis': {'range': [0, 1]}}
                    layout_updates.append(layout)

                elif plot_type == b'static_image':
                    data = data[0]
                    logger.debug('dashboard %s', data)
                    traces.append(go.Scatter())
                    image = {'source': data,
                             'xref': 'x', 'yref': 'y',
                             'x': 0, 'y': 1,
                             'sizex': 1, 'sizey': 1,
                             'sizing': "stretch",
                             'opacity': 1,
                             'layer': "above"}
                    images.append(image)

                    layout = {'xaxis': {'range': [0, 1], 'showgrid': False, 'zeroline': False,
                                        'showline': False, 'ticks': '', 'showticklabels': False},
                              'yaxis': {'range': [0, 1], 'showgrid': False, 'zeroline': False,
                                        'showline': False, 'ticks': '', 'showticklabels': False,
                                        'scaleanchor': 'x', 'scaleratio': 1.}}
                    layout_updates.append(layout)
                elif plot_type == b'text':
                    data = data[0]['pred']
                    ncols = len(data)
                    max_ntext = 0
                    for icol, dt in enumerate(data):
                        ntexts = len(dt)
                        max_ntext = max(ntexts, max_ntext)
                        trace = go.Scatter(
                            x=[icol] * ntexts,
                            y=list(range(ntexts)),
                            mode='text',
                            text=dt,
                            textposition='top center'
                        )
                        traces.append(trace)
                    layout = {
                        'xaxis': {
                            'range': [-0.5, ncols + 0.5],
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                            'ticks': '',
                            'showticklabels': False
                        },
                        'yaxis': {
                            'range': [max_ntext + 0.5, -0.5],
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                            'ticks': '',
                            'showticklabels': False,
                        },
                        'showlegend': False,
                        'font': {'size': 12}
                    }
                    layout_updates.append(layout)
                else:
                    warnings.warn('{} plot not implemented. Omitting this plot.'.format(plot_type))
            else:
                raise dash.exceptions.PreventUpdate()

        fig = go.Figure(traces)
        fig.layout.update({'autosize': True, 'title': ', '.join(titles)})

        if len(images) > 0:
            fig.layout.update({'images': images})

        for layout in layout_updates:
            fig.layout.update(layout)

        return fig

    return update_graph


def generate_refresh_selector():
    def refresh_selector(n):
        if n is not None:
            return get_data_options()

        else:
            raise dash.exceptions.PreventUpdate()

    return refresh_selector


def generate_add_graph(graph_index):
    def add_graph(n_graphs):
        if n_graphs == graph_index:
            return {'display': 'inline-block'}

        else:
            raise dash.exceptions.PreventUpdate()

    return add_graph


def generate_add_selector(graph_index):
    def add_selector(n_graphs):
        if n_graphs == graph_index:
            return {'display': 'inline-block'}

        else:
            raise dash.exceptions.PreventUpdate()

    return add_selector


max_n_graphs = 10

selector_list = []
graph_list = []
for graph_index in range(1, max_n_graphs + 1):
    selector = dcc.Dropdown(id=f'data-selector{graph_index}', value=[], options=[],
                            multi=True, className='data-selector', style={'display': 'none'})
    selector_list.append(selector)

    graph = dcc.Graph(id=f'graph{graph_index}', className='graph', style={'display': 'none'})
    graph_list.append(graph)


layout = [html.Button(u'↺', id='refresh-selector-button'),
          html.Button('+', id='add-graph-button'),
          html.Div(selector_list, id='data-selector-div'),
          html.Div(graph_list, id='graph-div'),
          html.Div('0', id='n-graphs', style={'display': 'none'}),
          dcc.Interval(id='interval-component', interval=1000, n_intervals=0)]


@app.callback(Output('n-graphs', 'children'),
              [Input('add-graph-button', 'n_clicks')],
              [State('n-graphs', 'children')])
def increment_graph_count(n, n_graphs):
    if n is None:
        raise dash.exceptions.PreventUpdate()

    else:
        n_graphs = int(n_graphs)
        n_graphs += 1
        n_graphs = str(n_graphs)

        return n_graphs


for graph_index in range(1, max_n_graphs + 1):
    graph_index = str(graph_index)
    app.callback(Output(f'data-selector{graph_index}', 'options'),
                 [Input('refresh-selector-button', 'n_clicks')])(generate_refresh_selector())

    app.callback(Output(f'graph{graph_index}', 'figure'),
                 [Input('interval-component', 'n_intervals'),
                  Input(f'data-selector{graph_index}', 'value')])(generate_update_graph())

    # make new graph (reveal existing graph)
    app.callback(Output(f'data-selector{graph_index}', 'style'),
                 [Input('n-graphs', 'children')])(generate_add_graph(graph_index))

    # make new graph (reveal existing graph)
    app.callback(Output(f'graph{graph_index}', 'style'),
                 [Input('n-graphs', 'children')])(generate_add_selector(graph_index))

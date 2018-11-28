import pickle
import warnings
import numbers
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


def make_mosaic(volume, x=None, y=None, z=None):
    if x is None:
        x = volume.shape[0] // 2
    if y is None:
        y = volume.shape[1] // 2
    if z is None:
        z = volume.shape[2] // 2
    return [go.Heatmap(z=volume[x, :, :], colorscale='Greys'),
            go.Heatmap(z=volume[:, y, :], colorscale='Greys'),
            go.Heatmap(z=volume[:, :, z], colorscale='Greys')]


layout = html.Div([html.H2("rtFMRI dashboard"),
                   html.Div([dcc.Dropdown(id='data-list', value='', multi=True)],
                            id='data-list-div'),
                   html.Div([dcc.Graph(id='graphs',
                                       style={'display': 'inline-block', 'height': '90%'})],
                            id='graph-div'),
                   dcc.Interval(id='interval-component', interval=500, n_intervals=0)],
                  style={'display': 'inline-block', 'width': '100vh', 'height': '100vh'},
                  className="container")

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
    for i, value in enumerate(selected_values):
        val = r.get(value)
        if val is not None:
            data = pickle.loads(val)
            plot_type = r.get(value + ':type')
            if plot_type == b'scatter':
                traces.append(go.Scatter(y=data))
                fig_specs.append([{'colspan': 3}, None, None])

            elif plot_type == b'bar':
                if isinstance(data, numbers.Number):
                    data = [data]
                traces.append(go.Bar(y=data))
                fig_specs.append([{'colspan': 3}, None, None])

            elif plot_type == b'timeseries':
                traces.append(go.Scatter(y=[1, 2, 3]))
                fig_specs.append([{'colspan': 3}, None, None])

            elif plot_type == b'mosaic':
                traces.append(make_mosaic(data))
                fig_specs.append([{}, {}, {}])

            else:
                warnings.warn('{} plot not implemented. Omitting this plot.'.format(plot_type))

    if len(traces) == 0:
        return go.Scatter()

    fig = plotly.tools.make_subplots(rows=len(selected_values), cols=3, specs=fig_specs,
                                     print_grid=False)

    for row_index, trace in enumerate(traces):
        if isinstance(trace, list):
            for column_index, tr in enumerate(trace):
                fig.append_trace(tr, row_index + 1, column_index + 1)
        else:
            fig.append_trace(trace, row_index + 1, 1)

    return fig

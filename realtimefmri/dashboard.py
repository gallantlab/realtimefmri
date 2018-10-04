import pickle
import warnings
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import flask
import redis

server = flask.Flask('app')
app = dash.Dash('app', server=server)

r = redis.Redis()

app.layout = html.Div([
    html.Div([dcc.Checklist(id='data-list', values=[],
                            labelStyle={'display': 'inline-block'}),
              html.Button('refresh', id='data-list-refresh-button')],
             id='menu'),
    dcc.Graph(id='graphs'),
    dcc.Interval(id='interval-component', interval=500, n_intervals=0)], className="container")


@app.callback(Output('data-list', 'options'),
              [Input('data-list-refresh-button', 'n_clicks'), Input('data-list', 'values')])
def update_data_list(n, values):
    return [{'label': key.lstrip(b'rt_'), 'value': key} for key in r.scan_iter()
            if not key.endswith(b'_type')]


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


@app.callback(Output('graphs', 'figure'),
              [Input('data-list', 'values'),
               Input('interval-component', 'n_intervals')])
def update_selected_graphs(selected_values, n):
    if len(selected_values) == 0:
        return go.Scatter()

    fig_specs = []
    traces = []
    for i, value in enumerate(selected_values):
        data = pickle.loads(r.get(value))
        plot_type = r.get(value + '_type')

        if plot_type == b'scatter':
            traces.append(go.Scatter(y=data))
            fig_specs.append([{'colspan': 3}, None, None])

        elif plot_type == b'mosaic':
            traces.append(make_mosaic(data))
            fig_specs.append([{}, {}, {}])

        else:
            warnings.warn('{} plot not implemented. Omitting this plot.'.format(plot_type))

    fig = plotly.tools.make_subplots(rows=len(selected_values), cols=3, specs=fig_specs,
                                     print_grid=False)

    for row_index, trace in enumerate(traces):
        if isinstance(trace, list):
            for column_index, tr in enumerate(trace):
                fig.append_trace(tr, row_index + 1, column_index + 1)
        else:
            fig.append_trace(trace, row_index + 1, 1)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

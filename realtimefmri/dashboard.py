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


# app.scripts.config.serve_locally = False
# dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

app.layout = html.Div([
    html.Div([dcc.Checklist(id='data-list', options=[], values=[],
                            labelStyle={'display': 'inline-block'}),
              html.Button('refresh', id='data-list-refresh-button')],
             id='menu'),
    html.Div([],
             id='graphs'),
    dcc.Interval(id='interval-component', interval=500, n_intervals=0)], className="container")


@app.callback(Output('data-list', 'options'),
              [Input('data-list-refresh-button', 'n_clicks'), Input('data-list', 'values')])
def update_data_list(n, values):
    return [{'label': key, 'value': key} for key in r.scan_iter()]


def infer_graph_type(data):
    pass


@app.callback(Output('graphs', 'children'),
              [Input('data-list-refresh-button', 'n_clicks'), Input('data-list', 'values')])
def update_selected_graphs(values):
    pass


@app.callback(Output('live-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    value = r.get('volume_data')
    if value is not None:
        value = np.frombuffer(value, dtype='float32')
        value = value.reshape(100, 100, 30)
        value = value[:, :, 15]
        print(value.shape)
        # value = value.mean((1, 2))
        # traces = [go.Scatter(x=np.arange(len(value)), y=value)]
        traces = [go.Heatmap(z=value)]
        fig = go.Figure(data=traces)
        # fig.layout['xaxis']['range'] = [0, 3]
        # fig.layout['yaxis']['range'] = [-10, 10]
        return fig


if __name__ == '__main__':
    app.run_server(debug=True)

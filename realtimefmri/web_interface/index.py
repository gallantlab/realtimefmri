import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from realtimefmri.web_interface.app import app
from realtimefmri.web_interface.apps import controls, dashboard, experiment, model, pipeline


def serve_layout():
    layout = html.Div([dcc.Location(id='url', refresh=False),
                       html.Div(id='page-content')])

    return layout


app.layout = serve_layout


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/controls':
        content = controls.layout

    elif pathname == '/dashboard':
        content = dashboard.layout

    elif pathname == '/pipeline':
        content = pipeline.layout

    elif pathname == '/':
        content = [html.A('controls', href='/controls'), html.Br(),
                   html.A('dashboard', href='/dashboard'), html.Br(),
                   html.A('experiments', href='/experiments'), html.Br(),
                   html.A('pipeline', href='/pipeline'), html.Br(),
                   html.A('viewer', href='http://localhost:8051')]
    else:
        content = '404'

    return content


def serve():
    app.run_server(host="0.0.0.0", debug=True)


if __name__ == '__main__':
    serve()

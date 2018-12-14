import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from realtimefmri.web_interface.app import app
from realtimefmri.web_interface.apps import controls, dashboard, pipeline


def serve_layout():
    layout = html.Div([dcc.Location(id='url', refresh=False),
                       html.Div(id='page-content')])

    return layout

app.layout = serve_layout

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/controls':
        return controls.layout

    elif pathname == '/dashboard':
        return dashboard.layout

    elif pathname == '/pipeline':
        return pipeline.layout

    else:
        return '404'


def serve():
    app.run_server(host="0.0.0.0", debug=False)


if __name__ == '__main__':
    serve()

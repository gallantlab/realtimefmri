import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from realtimefmri.web_interface.app import app
from realtimefmri.web_interface.apps import control_panel, dashboard, pipeline

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/control-panel':
        return control_panel.layout
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

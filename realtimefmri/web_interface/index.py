import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from realtimefmri.web_interface.app import app
from realtimefmri.web_interface.apps import control_panel, dashboard


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
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)

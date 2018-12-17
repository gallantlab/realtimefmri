import os
import os.path as op

import dash
import flask

external_stylesheets = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

css_directory = op.abspath(op.join(__file__, op.pardir, 'assets'))
stylesheets = ['style.css']
static_css_route = '/assets'


@app.server.route(f'{static_css_route}/<stylesheet>')
def serve_stylesheet(stylesheet):
    print(stylesheet)
    if stylesheet not in stylesheets:
        raise Exception(f'{stylesheet} is excluded from the allowed static files')
    return flask.send_from_directory(css_directory, stylesheet)


for stylesheet in stylesheets:
    app.css.append_css({'external_url': f'{static_css_route}/{stylesheet}'})

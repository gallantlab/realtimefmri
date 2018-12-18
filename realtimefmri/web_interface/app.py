import flask
import dash

from realtimefmri import config

external_stylesheets = []

app = dash.Dash(__name__, static_folder=config.STATIC_PATH,
                external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True


@app.server.route('/static/<path:path>')
def serve_static_file(path):
    return flask.send_from_directory(config.STATIC_PATH, path)


app.css.append_css({'external_url': f'{config.STATIC_PATH}/css/style.css'})

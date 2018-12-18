import dash
import flask

from realtimefmri import config
from realtimefmri.utils import get_logger

logger = get_logger('app', to_console=True)


external_stylesheets = []

app = dash.Dash(__name__, static_folder=config.STATIC_PATH,
                external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True


@app.server.route('/static/<path:path>')
def serve_static_file(path):
    logger.info(path)
    return flask.send_from_directory(config.STATIC_PATH, path)


app.css.append_css({'external_url': ['/static/css/style.css']})

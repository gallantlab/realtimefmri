import dash
import flask
import pickle
import redis
import time

from realtimefmri import config
from realtimefmri.utils import get_logger

logger = get_logger('app', to_console=True)


external_stylesheets = []

app = dash.Dash(__name__, static_folder=config.STATIC_PATH,
                external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True


r = redis.StrictRedis(config.REDIS_HOST)


@app.server.route('/experiment')
def serve_experiment():
    return flask.send_from_directory(config.STATIC_PATH, 'experiments/reaction_time.html')


@app.server.route('/redis/<key>')
def serve_redis(key):
    try:
        value = r.get(key)
        return pickle.loads(value)

    except Exception as e:
        logger.warning(e)


@app.server.route('/static/<path:path>')
def serve_static_file(path):
    logger.info(path)
    return flask.send_from_directory(config.STATIC_PATH, path)


app.css.append_css({'external_url': ['/static/css/style.css']})

import dash
import flask
import os.path as op
import pickle
import redis

from flask import render_template
from pathlib import Path

from realtimefmri import config
from realtimefmri.utils import get_logger

logger = get_logger('realtimefmri.app', to_console=True, to_file=True)


external_stylesheets = []

app = dash.Dash(__name__, static_folder=config.STATIC_PATH,
                external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True


r = redis.StrictRedis(config.REDIS_HOST)


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

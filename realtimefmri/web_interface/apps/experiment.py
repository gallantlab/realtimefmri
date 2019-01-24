import numpy as np
import pickle
import redis
import time

from flask import request
from pathlib import Path
from realtimefmri import config, utils
from realtimefmri.web_interface.app import app


logger = utils.get_logger(__name__)
r = redis.StrictRedis(config.REDIS_HOST)


def load_features(feature_name):
    """Load features for stimuli presented during the course of the experiment

    Read log entries posted to the database by the experiment client to find out which stimuli
    were presented and their start times. Load precomputed features for the stimuli.

    Parameters
    ----------
    feature_name : str

    Returns
    -------
    A list of arrays of feature times and a list of arrays of features
    """
    messages = []
    features = []
    feature_times = []
    for key in r.scan_iter('log:stimulus:*'):
        start_time = r.get(key + ':time')
        message = r.get(key + ':message')
        stimulus_name = message.split('start ')[1]
        feature_path = (Path(config.STATIC_PATH) / 'features' / feature_name /
                        Path(stimulus_name).with_suffix('.npy'))

        feat = np.load(feature_path)
        feat_times = np.arange(0, len(feat), dtype='float32') + start_time

        features.append(feat)
        feature_times.append(feat_times)

    return feature_times, features


def align_features_and_responses(feature_times, features, response_times, responses):
    return features, responses


@app.server.route('/models/<model_name>')
def serve_model(model_name, methods=['GET']):
    key_prefix = 'db:' + model_name
    keys = list(r.scan_iter(key_prefix + ':*'))
    return f'{len(keys)} time points for {key_prefix}'


@app.server.route('/models/<model_name>/fit', methods=['POST'])
def serve_fit_model(model_name):
    key_prefix = 'db:' + model_name
    response_times, responses = utils.load_timestamped_array_from_redis(key_prefix)

    # get feature times
    feature_times, features = load_features('motion_energy')

    X, y = align_features_and_responses(feature_times, features, response_times, responses)

    return f'Fitting {model_name}'


@app.server.route('/experiment/log/<topic>', methods=['POST'])
def serve_log(topic):
    """Store a log message from the client

    Messages must be posted as json containing the keys 'time' and 'message'

    Parameters
    ----------
    topic : str

    Returns
    -------
    HTTP status code
    """
    if request.method == 'POST':
        logger.debug(request.json)
        receive_time = time.time()
        r.set(f'log:{topic}:{receive_time}:time', request.json['time'])
        r.set(f'log:{topic}:{receive_time}:message', request.json['message'])

        return '200'


@app.server.route('/models/<model_name>/predict', methods=['POST'])
def serve_predict_model(model_name):
    return f'Predicting {model_name}'

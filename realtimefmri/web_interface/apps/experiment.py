import numpy as np
import pickle
import redis
import time
import urllib

from flask import request
from pathlib import Path
import scipy.stats
from sklearn import linear_model

from realtimefmri import config, detrend, utils
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


@app.server.route('/models')
def serve_models(methods='GET'):
    key_prefix = 'db:model:*'
    model_names = list(r.scan_iter(key_prefix))
    return b'Models:\n' + b'\n'.join(model_names)


@app.server.route('/models/<model_name>')
def serve_model(model_name, methods=['GET']):
    key_prefix = 'db:model:' + model_name
    keys = list(r.scan_iter(key_prefix + ':*'))
    return f'{len(keys)} time points for {key_prefix}'


@app.server.route('/models/<model_name>/fit', methods=['GET', 'POST'])
def serve_fit_model(model_name):
    query = urllib.parse.parse_qs(request.query_string.decode('utf-8').lstrip('?'))
    detrend_type = query.get('detrend_type', None)
    model_type = query.get('model_type', None)

    response_times, responses = utils.load_timestamped_array_from_redis('db:responses')

    if detrend_type == ['whitematter']:
        logger.warning('%s %d', detrend_type, len(response_times))
        _, wm_responses = utils.load_timestamped_array_from_redis('db:wm_responses')
        detrender = detrend.WhiteMatterDetrend(n_pcs=10)
        _ = detrender.fit(responses, wm_responses)
        responses = detrender.detrend(responses, wm_responses)

    # # get feature times
    # feature_times, features = load_features('motion_energy')
    # X, y = align_features_and_responses(feature_times, features, response_times, responses)

    y = scipy.stats.zscore(responses, axis=0)
    X = np.random.randn(len(y), 10).astype('float32')

    if model_type == ['ridge']:
        logger.warning('Fitting model %s', model_name)
        model = linear_model.LinearRegression()
        _ = model.fit(X, y)
        logger.warning('Saving model %s', model_name)
        r.set(f'db:model:{model_name}', pickle.dumps(model))

    return f'Fitting {model_name} {responses.shape}'


@app.server.route('/models/<model_name>/predict', methods=['GET'])
def serve_predict_model(model_name):
    query = urllib.parse.parse_qs(request.query_string.decode('utf-8').lstrip('?'))
    model_type = query.get('model_type', None)

    # # get feature times
    # feature_times, features = load_features('motion_energy')

    X = np.random.randn(20, 10).astype('float32')
    model = r.get(f'db:model:{model_name}')
    model = pickle.loads(model)
    y_hat = model.predict(X)

    return f'Predicting {model_name} {len(y_hat)}'


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

import pickle
from pathlib import Path

import numpy as np
import redis
from flask import render_template, request
from scipy import stats
from sklearn import linear_model

from realtimefmri import config, detrend, utils
from realtimefmri.web_interface.app import app


logger = utils.get_logger(__name__)
r = redis.StrictRedis(config.REDIS_HOST)


def load_responses(key_prefix, trials=None):
    """Load responses from the database

    Parameters
    ----------
    key_prefix : str
        Prefix to the key for the response in the database
    trials : list of int

    Returns
    -------
    An array of response times and an array of responses
    """
    if trials is None:
        response_times, responses = utils.load_timestamped_array_from_redis(key_prefix)

    else:
        responses, response_times = [], []
        for trial in trials:
            trial_key_prefix = f'{key_prefix}:trial{trial:04}'
            trial_response_times, trial_responses = load_responses(trial_key_prefix)
            response_times.append(trial_response_times)
            responses.append(trial_responses)

        response_times = np.concatenate(response_times, axis=0)
        responses = np.concatenate(responses, axis=0)

    return response_times, responses


def detrend_responses(key_prefix, detrend_type, trials=None):
    """Load and detrend responses

    Parameters
    ----------
    key_prefix : str
    detrend_type : str
    trials : list of int

    Returns
    -------
    An array with size (number of samples, number of voxels) of detrended responses
    """
    if detrend_type == 'whitematterdetrend':
        response_times, gm_responses = load_responses(key_prefix)
        _, wm_responses = load_responses('responses:whitematterdetrend')

        detrender = detrend.WhiteMatterDetrend()
        detrender.fit(gm_responses, wm_responses)

        if trials is not None:
            response_times, gm_responses = load_responses(key_prefix, trials=trials)
            _, wm_responses = load_responses('responses:whitematterdetrend', trials=trials)

        gm_detrended = detrender.detrend(gm_responses, wm_responses)

    else:
        raise NotImplementedError(f'{detrend_type} not implemented.')

    gm_detrended = stats.zscore(gm_detrended, axis=0)

    return response_times, gm_detrended


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


@app.server.route('/models', methods=['GET'])
def serve_models():
    """List all stored models, both those stored in the redis database and those stored on the
    server's file system. Models served on the file system can be loaded into the database.
    """
    datastore_models = []
    for model_path in (Path(config.DATASTORE_DIR) / 'models').glob('*.pkl'):
        datastore_models.append(model_path.stem)

    key_prefix = 'model:*'
    database_models = list(r.scan_iter(key_prefix))
    database_models = [m.decode('utf-8') for m in database_models]
    return render_template('models.html',
                           database_models=database_models,
                           datastore_models=datastore_models)


@app.server.route('/model/<model_name>', methods=['GET', 'POST'])
def serve_model(model_name):
    if request.method == 'GET':
        key = 'model:' + model_name
        model = r.get(key)
        if model is not None:
            model = pickle.loads(model)
            return f'{key} {str(model)}'
        else:
            return f'No model at {key}'


@app.server.route('/model/<model_name>/store', methods=['POST'])
def serve_store_model(model_name):
    with open(Path(config.DATASTORE_DIR) / f'models/{model_name}.pkl', 'rb') as f:
        r.set(f'model:{model_name}', f.read())

    return f'Stored model {model_name}'


@app.server.route('/model/<model_name>/fit', methods=['GET', 'POST'])
def serve_fit_model(model_name):
    """Fit a model

    parameters:
      - name: model_name
        in: path
        type: string
        description: Key for model in the database
      - name: mask_type
        in: query
        type: string
        description: Mask for responses
    """
    detrend_type = request.args.get('detrend_type', None)
    model_type = request.args.get('model_type', None)
    mask_type = request.args.get('mask_type', None)
    load_responses('responses:', )

    # # get feature times
    # feature_times, features = load_features('motion_energy')
    # X, y = align_features_and_responses(feature_times, features, response_times, responses)

    y = stats.zscore(responses, axis=0)
    X = np.random.randn(len(y), 10).astype('float32')

    if model_type == 'ridge':
        model = linear_model.LinearRegression()
        _ = model.fit(X, y)
        r.set(f'model:{model_name}', pickle.dumps(model))

    return f'Fitting {model_name} {responses.shape}'


@app.server.route('/model/<model_name>/decode', methods=['GET'])
def serve_decode_model(model_name):
    """Generate predictions from a model

    parameters:
      - name: model_name
        in: path
        type: string
        description: Key for model in the database
      - name: trials
        in: query
        type: list of integers
        description: Trials to decode from
    """
    # # get feature times
    # feature_times, features = load_features('motion_energy')
    trials = request.args.get('trials', None)

    load_responses()
    if trials is None:
        response_times, responses = utils.load_timestamped_array_from_redis('responses')

    X = np.random.randn(20, 10).astype('float32')
    model = r.get(f'model:{model_name}')
    model = pickle.loads(model)
    y_hat = model.predict(X)

    return f'Predicting {model_name} {len(y_hat)}'


@app.server.route('/model/<model_name>/predict', methods=['GET'])
def serve_predict_model(model_name):
    """Generate predictions from a model

    parameters:
      - name: model_name
        in: path
        type: string
        description: Key for model in the database
      - name: trials
        in: query
        type: list of integers
        description: Trials to use to make predictions
    """
    # # get feature times
    # feature_times, features = load_features('motion_energy')

    X = np.random.randn(20, 10).astype('float32')
    model = r.get(f'model:{model_name}')
    model = pickle.loads(model)
    y_hat = model.predict(X)

    return f'Predicting {model_name} {len(y_hat)}'

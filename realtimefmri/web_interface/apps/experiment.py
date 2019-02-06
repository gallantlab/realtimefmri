import datetime
import json
import pickle
import time
from pathlib import Path

import numpy as np
import redis
from flask import render_template, request, Response, send_from_directory

from realtimefmri import config, utils
from realtimefmri.web_interface.app import app
from realtimefmri.web_interface.apps.model import detrend_responses


logger = utils.get_logger(__name__)
r = redis.StrictRedis(config.REDIS_HOST)


@app.server.route('/experiment/run/<path:path>')
def serve_run_experiment(path):
    return send_from_directory(config.EXPERIMENT_DIR, path + '.html')


@app.server.route('/experiments')
def serve_experiments():
    experiments = Path(config.EXPERIMENT_DIR).glob('*.html')
    experiment_names = [e.stem for e in experiments]
    return render_template('experiments.html', experiment_names=experiment_names)


@app.server.route('/experiment/trial/append/random', methods=['POST'])
def serve_append_random_stimulus_trial():
    """Predict the optimal (and minimal) stimuli

    parameters:
      - name: n
        in: query
        type: integer
        description: Number of random stimuli to append
    """
    n = request.args.get('n', None)
    if n is None:
        n = 2
    else:
        n = int(n[0])

    for i in range(n):
        trial = {'type': 'video',
                 'sources': [f'/static/videos/{i + 1:02}.mp4'],
                 'width': 400, 'height': 400, 'autoplay': True}
        r.lpush('experiment:trials', pickle.dumps(trial))

    return f'Appending {n} random trials'


@app.server.route('/experiment/trial/append/top_n', methods=['POST'])
def serve_append_top_n():
    """Add the top n most likely decoding results

    parameters:
      - name: model_name
        in: query
        type: string
        required: true
        description: Name of a pre-trained model. Must exist in the database as a pickled Python
            class with a ``predict_proba`` method that outputs a score for each class and a
            ``class_names`` attribute containing string representations of the classes.
      - name: n
        in: query
        type: integer
        description: Number of random stimuli to append
      - name: detrend_type
        in: query
        type: string
        description: Type of detrending to apply
    """
    model_name = request.args.get('model_name')
    responses_name = request.args.get('responses_name', 'graymatter')
    n = int(request.args.get('n', '5'))
    detrend_type = request.args.get('detrend_type', 'whitematterdetrend')

    model = pickle.loads(r.get(f'model:{model_name}'))
    trial_index = pickle.loads(r.get('experiment:trial:current'))['index']

    key_prefix = f'responses:{responses_name}'
    _, responses = detrend_responses(key_prefix, detrend_type=detrend_type, trials=[trial_index])

    probabilities = model.predict_proba(responses.mean(0, keepdims=True)).ravel()
    top_indices = probabilities.argsort()[-n:][::-1]
    top_sizes = 10 + np.arange(n) * 10
    top_class_names = model.class_names[top_indices]

    stimulus = ''
    for size, class_name in zip(top_sizes[::-1], top_class_names[::-1]):
        stimulus += f"<p style='font-size:{size}px'>{class_name}</p>"

    trial = {'type': 'html-keyboard-response',
             'stimulus': stimulus,
             'stimulus_duration': 2000,
             'trial_duration': 2000}

    r.lpush('experiment:trials', pickle.dumps(trial))

    return f'Appending top {n} trial'


@app.server.route('/experiment/trial/append/optimal_stimuli', methods=['POST'])
def serve_append_optimal_stimulus_trial():
    """Predict the optimal (and minimal) stimuli

    parameters:
      - name: model_name
        in: query
        type: string
        required: true
        description: Name of a pre-trained model
      - name: n_optimal
        in: query
        type: integer
        description: Number of optimal stimuli to append
      - name: n_minimal
        in: query
        type: integer
        description: Number of minimal stimuli to append
      - name: n_random
        in: query
        type: integer
        description: Number of random stimuli to append
    """
    model_name = request.args.get('model_name')
    if model_name is None:
        return 'Must provide model_name in query string.'

    n_optimal = request.args.get('n_optimal', 3)
    n_minimal = request.args.get('n_minimal', 3)
    n_random = request.args.get('n_random', 3)

    X = np.random.randn(5, 10).astype('float32')
    model = r.get(f'model:{model_name}')
    model = pickle.loads(model)
    y_hat = model.predict(X)

    optimal_indices = y_hat.argpartition(-n_optimal)[-n_optimal:]
    minimal_indices = y_hat.argpartition(n_minimal)[:n_minimal]

    for i in range(1, 5):
        trial = {'type': 'video',
                 'sources': [f'/static/videos/{i:02}.mp4'],
                 'width': 400, 'height': 400, 'autoplay': True}
        r.lpush('experiment:trials', pickle.dumps(trial))

    return f'Predicting {model_name} {len(y_hat)}'


@app.server.route('/experiment/trial/next', methods=['POST'])
def serve_next_trial():
    trial = r.rpop('experiment:trials')
    if trial is None:
        return 'No trials remaining'

    trial = pickle.loads(trial)
    return json.dumps(trial)


@app.server.route('/experiment/trial/new', methods=['POST'])
def serve_new_trial():
    """Initialize a new trial"""
    start_time = request.args.get('time')

    current_trial = r.get('experiment:trial:current')
    if current_trial is None:
        trial_index = 0
    else:
        previous_trial = pickle.loads(current_trial)
        trial_index = previous_trial['index'] + 1

    current_trial = {'start_time': start_time, 'end_time': None, 'index': trial_index}
    trial = pickle.dumps(current_trial)
    r.set('experiment:trial:current', trial)
    r.set(f'experiment:trial:{trial_index}', trial)
    return json.dumps(current_trial)


@app.server.route('/experiment/trial/current', methods=['GET'])
def serve_current_trial():
    """Trial number

    parameters:
      - name: model_name
        in: query
        type: string
        required: true
        description: Name of a pre-trained model
      - name: n_optimal
        in: query
        type: integer
        description: Number of optimal stimuli to append
      - name: n_minimal
        in: query
        type: integer
        description: Number of minimal stimuli to append
      - name: n_random
        in: query
        type: integer
        description: Number of random stimuli to append
    """
    current_trial = r.get('experiment:trial:current')
    if current_trial is None:
        response = 'No trial currently active'

    else:
        current_trial = pickle.loads(current_trial)

        if request.method == 'GET':
            response = json.dumps(current_trial)

    return response


@app.server.route('/experiment/trial/reset', methods=['POST'])
def serve_reset_trial():
    """Reset the trial count"""
    for key in r.scan_iter('experiment:trial:*'):
        r.delete(key)

    return f'Trial count reset'


@app.server.route('/experiment/logs', methods=['GET'])
def serve_experiment_logs():
    """Reset the trial count"""
    logs = []
    for key in r.scan_iter('experiment:log:*'):
        log = pickle.loads(r.get(key))
        logs.append({'name': key.decode('utf-8').split(':')[-1], 'length': len(log)})

    return render_template('logs.html', logs=logs)


@app.server.route('/experiment/log/store', methods=['POST'])
def serve_store_experiment_log():
    """Store the experiment log"""
    log = json.loads(request.data)
    log_name = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    key = f'experiment:log:{log_name}'
    r.set(key, pickle.dumps(log))
    return f'Saved log to {log_name}'


@app.server.route('/experiment/log/<log_name>/download', methods=['GET'])
def serve_download_experiment_log(log_name):
    """Send download of experiment log"""
    key = f'experiment:log:{log_name}'
    log = r.get(key)
    if log is None:
        response = f'No log found for {log_name}'

    else:
        log = pickle.loads(log)
        log = json.dumps(log)
        response = Response(log, mimetype="application/json",
                            headers={"Content-disposition": f"attachment; filename={log_name}.json"})

    return response


@app.server.route('/log/<topic>', methods=['POST'])
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
        receive_time = time.time()
        r.set(f'log:{topic}:{receive_time}:time', request.json['time'])
        r.set(f'log:{topic}:{receive_time}:message', request.json['message'])

        return '200'

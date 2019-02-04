import json
import pickle
import time
from pathlib import Path

import numpy as np
import redis
from flask import render_template, request, send_from_directory

from realtimefmri import config, utils
from realtimefmri.web_interface.app import app


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
        description: Name of a pre-trained model
      - name: n
        in: query
        type: integer
        description: Number of random stimuli to append
    """
    model_name = request.args.get('model_name')
    n = int(request.args.get('n', '5'))

    model = pickle.loads(r.get(f'db:model:{model_name}'))
    concept_names = 'a,absorption,acknowledgeable,adjudicate,affect,Aida,alewife,Almaden,Amerada,anastomosis,anniversary,any,apprise,Argonne,artifice,Assyriology,attribute,avalanche,background,Bamako,barracuda,bawd,befall,Benedikt,bestir,billet,Blackburn,blot,Bolshevism,boson,brakeman,Bridgetown,browse,bullfrog,busload,Caesarian,camouflage,capstan,carousel,catenate,centenary,channel,chemisorb,chlorine,Churchillian,Clarence,close,codeposit,colloidal,commune,concentrate,Confucius,constant,controller,Corinth,cotman,Cowan,creekside,CRT,cupric,cytochemistry,Dartmouth,decal,deficient,demark,depressant,deterring,Dickinson,diocesan,dissemble,doghouse,doublet,dressmake,Dumpty,earthenware,eerie,electroencephalogram,embedded,endogamy,epicyclic,erosive,eucalyptus,exasperate,expansible,extramarital,fallen,fearsome,fetish,fingerprint,flash,flu,footstep,foster,freeman,Fuchs,gag,garland,genesis,gibbon,glean,Godwin,graham,Greg,GSA,Gwyn,Halstead,Harcourt,havoc,hedgehog,heredity,hidalgo,ho,homologue,Horus,humility,hymen,Ifni,impedance,inadvertent,incorrect,Indus,inflow,inputting,intemperance,invasive,irrepressible,jackknife,Jesus,joy,kaiser,keto,Klein,Kurd,landlord,laureate,leftward,Levin,limelight,Litton,Lombardy,lubricious,Lyman,magnanimity,Malta,Marceau,Maryland,Maya,meaty,Mendelssohn,Metcalf,might,minstrelsy,Moe,monstrosity,motet,multiplicity,myopic,nawab,nervous,niggardly,nondescript,nu,oblong,off,omnivore,orchestra,ostrich,pagoda,paperback,parry,patriot,pediatric,percent,personage,phi,picojoule,piss,playa,poesy,pompadour,postcondition,precious,primitive,prokaryotic,protagonist,psyllium,purpose,quantile,r,ramshackle,reason,reflexive,Rene,resultant,rhombus,rivet,rosebud,ruination,safekeeping,sanctuary,Saudi,scherzo,scrappy,sebaceous,Semitic,settle,shaven,shoemake,sibyl,simile,sketchy,slippery,snapdragon,solder,sorption,specific,splenetic,squawk,stark,Sternberg,stormbound,Stuart,sudden,superfluity,swain,syllabic,tactic,Tarbell,teet,term,Theodore,thorough,tide,tofu,torrential,transcendent,treat,tritium,tularemia,twirl,unilateral,urinate,vampire,Venusian,video,vivid,Wahl,Washington,wee,wherein,wigging,wireman,Wordsworth,xenon,your'
    concept_names = concept_names.split(',')
    probabilities = np.random.randn(len(concept_names))

    top_indices = probabilities.argsort()[-n:][::-1]
    top_sizes = 10 + np.arange(n) * 10
    logger.warning(top_indices)
    logger.warning(top_sizes)
    logger.warning(concept_names)

    stimulus = ''
    for i in range(n):
        stimulus += f"<p style='font-size:{top_sizes[i]}px'>{concept_names[top_indices[i]]}</p>"

    trial = {'type': 'html-keyboard-response',
             'stimulus': stimulus,
             'stimulus_duration': 2000,
             'trial_duration': 2000}

    logger.warning('stimulus %s', str(trial))
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
    model = r.get(f'db:model:{model_name}')
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

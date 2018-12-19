"""Utilities for reading, writing, and managing pipelines"""
import importlib
import inspect
import re
from collections import OrderedDict

import yaml


def load_class(absolute_class_name):
    """Import a class or function from a string

    Parameters
    ----------
    absolute : str
        Absolute import name, i.e. realtimefmri.preprocess.Debug

    Returns
    -------
    A class
    """
    module_name, class_name = absolute_class_name.rsplit('.', maxsplit=1)

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_step_name(step):
    """Get the full name of a step's python class, e.g., realtimefmri.preprocess.ApplyMask
    """
    patt = re.compile(r"<class '(?P<class_name>[A-Za-z0-9\.\_]*)'>")
    class_name = patt.match(str(step)).groupdict()['class_name']
    return class_name


def get_instance_parameters(inst):
    """Get the parameters of a
    """
    parameters = []
    for at in inspect.getmembers(inst):
        if not at[0].startswith('__') and at[1]:
            parameters.append(at)
    return parameters


def get_init_parameters(step):
    """Get a dict of the names and default values required to initialize a step
    """
    signature = inspect.signature(step)
    parameters = OrderedDict()
    for k in signature.parameters:
        p = signature.parameters[k]
        if p.default == inspect._empty:
            default = '<value>'
        else:
            default = p.default

        parameters.update({p.name: default})

    return parameters


def get_run_parameters(step):
    """Get a dict of the names of values require to run a step
    """
    run_signature = inspect.signature(step.run)
    parameters = []
    for p in run_signature.parameters:
        if p != 'self':
            parameters.append(p)
    return parameters


def generate_pipeline(steps):
    """Generate a pipeline from a list of steps
    """
    pipeline = {'pipeline': []}
    for step in steps:
        class_name = get_step_name(step.__class__)
        init_params = get_init_parameters(step)
        run_params = get_run_parameters(step)
        p = OrderedDict()
        p.update({'name': class_name})
        p.update({'kwargs': init_params})
        p.update({'input': run_params})
        p.update({'output': []})

        pipeline['pipeline'].append(p)

    return pipeline


def save_pipeline_template(steps, path):
    """Create a pipeline template for a given set of preprocessing steps

    Parameters
    ----------
    steps : list of preprocessing steps
    path : str

    Returns
    -------
    A yaml-formatted string for the pipeline template
    """
    def setup_yaml():
        """ http://stackoverflow.com/a/8661021 """
        def represent_ordered_dict(self, data):
            return self.represent_mapping('tag:yaml.org,2002:map', data.items())

        yaml.add_representer(OrderedDict, represent_ordered_dict)

    setup_yaml()

    template = yaml.dump(generate_pipeline(steps))
    with open(path, 'w') as f:
        f.write(template)

    return template

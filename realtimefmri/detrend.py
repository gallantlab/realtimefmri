import six
if six.PY2:
    import cPickle as pickle
elif six.PY3:
    import pickle

import os.path as op
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from realtimefmri.config import get_subject_directory


class WhiteMatterDetrending(object):
    def __init__(self, n_wm_pcs=10):
        self.n_wm_pcs = n_wm_pcs
        self.model = None
        self.pca = None

    def train(self, gm, wm, n_wm_pcs=10):
        n_trials, _ = wm.shape

        pca = PCA(n_components=n_wm_pcs)
        wm_pcs = pca.fit_transform(wm)

        model = LinearRegression()
        model.fit(wm_pcs, gm)

        self.model = model
        self.pca = pca

    def detrend(self, gm, wm):
        trend = self.model.predict(self.pca.transform(wm))
        return gm - trend

    def save(self, subject, name):
        subj_dir = get_subject_directory(subject)
        with open(op.join(subj_dir, 'model-{}.pkl'.format(name)), 'r') as f:
            pickle.dump(self.model, f)
        with open(op.join(subj_dir, 'pca-{}.pkl'.format(name)), 'r') as f:
            pickle.dump(self.pca, f)

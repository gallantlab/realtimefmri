import os.path as op
import pickle

from sklearn import decomposition, linear_model

from realtimefmri.config import get_subject_directory


class WhiteMatterDetrend():
    def __init__(self, n_pcs=10):
        self.n_pcs = n_pcs
        self.model = None
        self.pca = None

    def fit(self, gm, wm, n_pcs=10):
        pca = decomposition.PCA(n_components=n_pcs)
        pcs = pca.fit_transform(wm)

        model = linear_model.LinearRegression()
        model.fit(pcs, gm)

        self.model = model
        self.pca = pca
        return self

    def detrend(self, gm, wm):
        trend = self.model.predict(self.pca.transform(wm))
        return gm - trend

    def save(self, subject, name):
        subj_dir = get_subject_directory(subject)
        with open(op.join(subj_dir, 'model-{}.pkl'.format(name)), 'r') as f:
            pickle.dump(self.model, f)
        with open(op.join(subj_dir, 'pca-{}.pkl'.format(name)), 'r') as f:
            pickle.dump(self.pca, f)

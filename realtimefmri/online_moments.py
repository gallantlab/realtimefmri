"""
Compute online moments

Anwar O. Nunez-Elizalde (anwarnunez@berkeley.edu)
"""

import numpy as np
from realtimefmri.preprocess import PreprocessingStep


class OnlineMoments(PreprocessingStep):
    """Compute 1-Nth raw moments online

    For the Ith moment: E[X^i] = (1/n)*\Sum(X^i). This function only stores
    \Sum(X^i) and keeps track of the number of observations.

    Parameters
    ----------
    order : int
        The number of moments to compute

    Attributes
    ----------
    order : int
        The number of moments to compute
    all_raw_moments : numpy.ndarray
        All of the raw moments

    Methods
    -------
    update(x)
        Update the moments given the new observations
    get_statistics()
        Compute the statistics for the data
    get_raw_moments()
        Return the raw moments
    get_norm_raw_moments
        Return normalized raw moments
    run(inp)
        Return the mean and standard deviation
    """
    def __init__(self, order=4, **kwargs):
        self.n = 0.0
        self.order = order
        self.all_raw_moments = [0.0]*self.order
        for odx in range(self.order):
            self.__setattr__('rawmnt%i'%(odx+1), self.all_raw_moments[odx])

    def __repr__(self):
        return '%s.online_moments' % (__name__)

    def update(self, x):
        """Update the raw moments

        Parameters
        ----------
        x : np.ndarray, or scalar-like
            The new observation. This can be any dimension.
        """
        self.n += 1

        for odx in range(self.order):
            name = 'rawmnt%i' % (odx + 1)
            self.all_raw_moments[odx] = self.all_raw_moments[odx] + x**(odx + 1)
            self.__setattr__(name, self.all_raw_moments[odx])

    def get_statistics(self):
        """Return the 1,2,3,4-moment estimates"""
        # mean,var,skew,kurt
        return convert_parallel2moments([self.get_raw_moments()[:4]],
                                        self.n)

    def get_raw_moments(self):
        return self.all_raw_moments

    def get_norm_raw_moments(self):
        return map(lambda x: x/float(self.n), self.all_raw_moments)

    def run(self, inp):
        self.update(inp)
        return self.get_statistics()[:2]


def compute_raw2var(raw1, raw2, *args):
    """Use the raw moments to compute the 2nd central moment
    VAR(X) = E[X^2] - E[X]^2
    """
    return raw2 - raw1**2


def compute_raw2skew(raw1, raw2, raw3, *args):
    """Use the raw moments to compute the 3rd standardized moment
    Skew(X) = (E[X^3] - 3*E[X]*E[X^2] + 2*E[X]^3)/VAR(X)^(3/2)
    """
    # get central moments
    cm2 = raw2 - raw1**2
    cm3 = raw3 - 3 * raw1 * raw2 + 2 * raw1**3
    # get standardized 3rd moment
    sm3 = cm3 / cm2**1.5
    return sm3


def compute_raw2kurt(raw1, raw2, raw3, raw4, *args):
    """Use the raw moments to compute the 4th standardized moment
    Kurtosis(X) = (E[X^4] - 4*E[X]*E[X^3] + 6*E[X]^2*E[X^2] - 3*E[X]^4)/VAR(X)^2 - 3.0
    """
    # get central moments
    cm2 = raw2 - raw1**2
    cm4 = raw4 - 4 * raw1 * raw3 + 6 * (raw1**2) * raw2 - 3 * raw1**4
    # get standardized 4th moment
    sm4 = cm4 / cm2**2 - 3
    return sm4


def convert_parallel2moments(node_raw_moments, nsamples):
    """Combine the online parallel computations of
    `online_moments` objects to compute moments.

    Parameters
    -----------
    node_raw_moments : list
        Each element in the list is a node output.
        Each node output is a `len(4)` object where the
        Nth element is Nth raw moment sum: `np.sum(X**n)`.
        Node moments are aggregated across nodes to compute:
        mean, variance, skewness, and kurtosis.
    nsamples : scalar
        The total number of samples across nodes


    Returns
    -------
    mean : array-like
        The mean of the full distribution
    variance: array-like
        The variance of the full distribution
    skewness: array-like
        The skewness of the full distribution
    kurtosis: array-like
        The kurtosis of the full distribution
    """
    mean_moments = []
    for raw_moment in zip(*node_raw_moments):
        moment = np.sum(raw_moment, 0) / nsamples
        mean_moments.append(moment)

    emean = mean_moments[0]
    evar = compute_raw2var(*mean_moments)
    eskew = compute_raw2skew(*mean_moments)
    ekurt = compute_raw2kurt(*mean_moments)
    return emean, evar, eskew, ekurt

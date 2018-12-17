import numpy as np

from realtimefmri import buffered_array, preprocess


class InvGammaParameters(object):
    def __init__(self, alpha=None, beta=None, variance=None):
        self.alpha = alpha
        self.beta = beta
        self.variance = variance

    def get_alpha(self):
        return 1 + (self.beta / self.variance)

    def get_beta(self):
        return self.variance * (self.alpha + 1)

    def get_variance(self):
        return self.beta / (self.alpha + 1)


class BayesianZScore(preprocess.PreprocessingStep):
    def __init__(self, prior_means, prior_variances,
                 mean_belief, variance_alpha, update_prior=True):
        """Preprocessing module that z-scores data using mean and variance estimated
        using Bayesian methods

        Attributes
        ----------
        prior_means : numpy.ndarray
            A vector containing the mean prior values for each feature
        prior_variances : numpy.ndarray
            A vector containing the variance prior values for each features
        mean_belief : float or numpy.ndarray
            A float or vector of floats of the ratio determining how much to weigh
            the prior over the data
        variance_alpha : float or numpy.ndarray
            A float or vector of floats containing the alpha parameter for the
            prior distribution on the variance
        update_prior : bool
            Flag indicating whether to update the prior with the posterior after each
            input value
        """
        inverse_gamma = InvGammaParameters(None, None, prior_variances)
        inverse_gamma.alpha = variance_alpha
        inverse_gamma.beta = inverse_gamma.get_beta()

        self.prior_means = prior_means
        self.prior_variances = prior_variances
        self.mean_belief = mean_belief
        self.inverse_gamma = inverse_gamma
        self.update_prior = update_prior

    def run(self, inp):
        """Run the z-scoring on one time point and update the prior

        Parameters
        ----------
        inp : numpy.ndarray
            A vector of data to be z-scored

        Returns
        -------
        The input array z-scored using the posterior mean and variance
        """
        if not hasattr(self, 'data'):
            self.data = buffered_array.BufferedArray(inp.size, dtype=inp.dtype)

        self.data.append(inp)

        post_var = compute_posterior_variance(self.data.get_array(),
                                              self.prior_means,
                                              self.inverse_gamma.alpha,
                                              self.inverse_gamma.beta)
        post_mean = compute_posterior_mean(self.data.get_array(),
                                           self.prior_means, self.mean_belief)

        if self.update_prior:
            self.prior_means = post_mean
            self.prior_variances = post_var

        return (inp - post_mean) / np.sqrt(post_var)

    def reset(self):
        del self.data


def compute_posterior_variance(x, prior_mean, alpha, beta, axis=0):
    """Compute the variance of a vector given a prior variance

    Parameters
    ----------
    x : numpy.ndarray
        Vector or array to compute the variance on
    prior_mean : float of numpy.ndarray
        Prior mean
    alpha : float
        Alpha parameter of the inverse gamma prior
    beta : float
        Beta parameter of the inverse gamma prior
    axis : int, optional
        For data arrays, the dimension to compute the variance along
    """
    N = len(x)
    v1 = 1. / (N + 2 * alpha + 2)
    v2 = 2. * beta + np.sum((x - prior_mean)**2, axis)
    variance = v1 * v2
    return variance


def compute_posterior_mean(x, prior_mean, belief=10., axis=0):
    """Compute the mean of a vector given a prior mean

    Parameters
    ----------
    x : numpy.ndarray
        Vector or array to compute the variance on
    prior_mean : float of numpy.ndarray
        Prior mean
    belief : float
        Ratio determining how much to weigh the prior over the data
    beta : float
        Beta parameter of the inverse gamma prior
    axis : int, optional
        For data arrays, the dimension to compute the mean along
    """
    N = len(x)
    m1 = 1. / (N + belief)
    m2 = np.sum(x, axis) + belief * prior_mean
    mean = m1 * m2
    return mean

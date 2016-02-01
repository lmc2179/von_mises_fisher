import numpy as np
from math import exp, pi, log
from scipy.special import iv
import random
from collections import Counter

def vmf_pdf(x, mu, kappa):
    """
    The pdf of the <a href=https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>von Mises-Fisher distribution</a>.

    Parameters:
        mu: float, location parameter of the distribution. This is the mean and mode of the distribution.
        kappa: positive float, scale parameter of the distribution. A larger value of kappa corresponds to lower variance.
    Returns:
        float, value of VMF(x|mu, kappa)
    """
    p = len(x)
    likelihood = _get_vmf_likelihood_term(x, mu, kappa)
    normalization_numerator = _get_vmf_normalization_numerator(p, kappa)
    normalization_denominator = _get_vmf_normalization_denom(p, kappa)
    return likelihood * (normalization_numerator / normalization_denominator)

def vmf_log_pdf(x, mu, kappa):
    """
    The log pdf of the <a href=https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>von Mises-Fisher distribution</a>.

    Parameters:
        mu: float, location parameter of the distribution. This is the mean and mode of the distribution.
        kappa: positive float, scale parameter of the distribution. A larger value of kappa corresponds to lower variance.
    Returns:
        float, value of log VMF(x|mu, kappa)
    """
    p = len(x)
    likelihood = _get_vmf_likelihood_term(x, mu, kappa)
    normalization_numerator = _get_vmf_normalization_numerator(p, kappa)
    normalization_denominator = _get_vmf_normalization_denom(p, kappa)
    return log(likelihood) + log(normalization_numerator) - log(normalization_denominator)

def _get_vmf_likelihood_term(x, mu, kappa):
    return exp(kappa * np.dot(mu, x))

def _get_vmf_normalization_numerator(p, kappa):
    return kappa ** (0.5*p - 1)

def _get_vmf_normalization_denom(p, kappa):
    return (2 * pi) ** (0.5*p) * iv(0.5*p-1, kappa)

def vmf_mixture_log_pdf(x, estimated_params, mixing):
    pdf_values = [vmf_pdf(x, *p) for p in estimated_params]
    return log(sum([v * m for v,m in zip(pdf_values, mixing)]))

def vmf_mle(X):
    """
    Calculate the maximum likelihood estimate of the parameters for the vMF distribution on the data.

    Parameters:
        X: numpy array, matrix of observed unit vectors
    Returns:
        mle_mu: MLE of mu calculated from the data
        mle_kappa: Approximation to MLE of kappa, using the fast approximation given in [0]

    [0] A. Banerjee, I. S. Dhillon, J. Ghosh, and S. Sra. Clustering on the Unit Hypersphere using von Mises-Fisher Distributions. JMLR, 6:1345–1382, Sep 2005
    """
    sum_inputs = sum(X)
    p = len(X[0])
    n = len(X)
    sum_norm = np.linalg.norm(sum_inputs)
    centroid = sum_inputs/sum_norm
    R = sum_norm / n
    mle_kappa = (R * (p - R**2)) / (1 - R**2)
    return centroid, mle_kappa

def vmf_mixture_mle(X, n, iterations):
    """
    Calculate the maximum likelihood estimate of a mixture of von Mises-Fisher distributions on your data using the EM algorithm.

    Parameters:
        X: The data on which to compute the MLE.
        n: The number of components of the mixture
        iterations: The number of iterations to run the EM algorithm.
    Returns:
        params: list of tuples, pairs of (mu, kappa) for each distribution.
        mixing: list of floats which add to 1.0, the mixing coefficients for each distribution
    """
    if len(X) < n:
        raise Exception('Fewer components than data points')
    params = _get_initial_mixture_params(X, n)
    Z = _assign_mle_Z(X, params)
    for i in range(iterations):
        params = _get_mixture_mle_params(X, Z, n)
        Z = _assign_mle_Z(X, params)
    Z_counter = Counter(Z)
    mixing = [Z_counter[k] / len(Z) for k in sorted(Z_counter.keys())]
    return params, mixing

def _get_initial_mixture_params(X, n):
    indices = list(range(len(X)))
    random.shuffle(indices)
    initial_mu_indices = indices[:n]
    initial_mu_values = [X[i] for i in initial_mu_indices]
    return [(mu, 20) for mu in initial_mu_values]

def _assign_mle_Z(X, params):
    Z = []
    for x in X:
        likelihoods = [vmf_pdf(x, *p) for p in params]
        Z.append(np.argmax(likelihoods))
    return Z

def _get_mixture_mle_params(X, Z, n):
    samples_by_mixture = [list() for i in range(n)]
    for z, x in zip(Z, X):
        samples_by_mixture[z].append(x)
    mle_params = [vmf_mle(samples) for samples in samples_by_mixture]
    return mle_params

def fit_vmf_mixture_BIC(X, possible_component_counts, em_iterations):
    """
    Fit a number of mixture distributions with the EM algorithm, and select the optimal model using the Bayesian Information Criterion.

    Parameters:
        X: The data used to fit the models
        possible_component_counts: iterable of integers; each integer will be used to fit a model with that many components. For example, if this value is [2, 3], a model with either two or three components will be returned
        em_iterations: Number of iterations to run the EM algorithm when computing the MLE for each component.
    Returns:
        params: The parameters of the optimal mixture model under the BIC
        best_component_choice: This will be one of the elements of possible_component_counts, corresponding to the optimal model
    """
    BICs = []
    mle_params = []
    for n_components in possible_component_counts:
        score, params = _evaluate_model_BIC(X, n_components, em_iterations)
        BICs.append(score)
        mle_params.append(params)
    best_model_index = np.argmin(BICs)
    return mle_params[best_model_index], possible_component_counts[best_model_index]

def _evaluate_model_BIC(X, n_components, em_iterations):
    estimated_params, mixing = vmf_mixture_mle(X, n_components, em_iterations)
    ln_L = sum([vmf_mixture_log_pdf(x, estimated_params, mixing) for x in X])
    k = 2 * n_components
    n = len(X)
    score = -2 * ln_L + k * log(n)
    return score, estimated_params
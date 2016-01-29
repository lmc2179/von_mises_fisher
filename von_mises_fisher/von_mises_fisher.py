import numpy as np
from math import exp, pi
from scipy.special import iv

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
    likelihood = exp(kappa * np.dot(mu, x))
    normalization_numerator = kappa ** (0.5*p - 1)
    normalization_denominator = (2 * pi) ** (0.5*p) * iv(0.5*p-1, kappa)
    return likelihood * (normalization_numerator / normalization_denominator)

def vmf_mle(X):
    """
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
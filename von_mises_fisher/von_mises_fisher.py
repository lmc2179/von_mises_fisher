import numpy as np
from math import exp, pi
from scipy.special import iv

def vmf_pdf(x, mu, kappa):
    p = len(x) + 1
    likelihood = exp(kappa * np.dot(mu, x))
    normalization_numerator = kappa ** (p/2 - 1)
    normalization_denominator = 2 * pi ** (p/2) * iv(p/2-1, kappa)
    return likelihood * (normalization_numerator / normalization_denominator)
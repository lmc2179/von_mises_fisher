import numpy as np
from math import exp, pi
from scipy.special import iv

def vmf_pdf(x, mu, kappa):
    p = len(x) + 1
    likelihood = exp(kappa * np.dot(mu, x))
    normalization_numerator = kappa ** (0.5*p - 1)
    normalization_denominator = 2 * pi ** (0.5*p) * iv(0.5*p-1, kappa)
    return likelihood * (normalization_numerator / normalization_denominator)
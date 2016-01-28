import unittest
import numpy as np
from von_mises_fisher.von_mises_fisher import vmf_pdf
from math import cos, sin, acos, asin, sqrt, pi
from matplotlib import pyplot as plt

RIGHT_ANGLE = pi / 2

def angle_to_vector(theta):
    return np.array([cos(theta), sin(theta)])

class PdfTest(unittest.TestCase):
    def test_2d_radial_symmetry(self):
        mu_angle = np.random.uniform(0, RIGHT_ANGLE)
        mu = angle_to_vector(mu_angle)
        kappa = 0.5
        max_angle = min(acos(mu[0]), RIGHT_ANGLE - acos(mu[0]))
        angle_perturb = np.random.uniform(0, max_angle)
        angle_1 = max_angle + angle_perturb
        angle_2 = max_angle - angle_perturb
        x1 = np.array([cos(angle_1), sin(angle_1)])
        x2 = np.array([cos(angle_2), sin(angle_2)])
        print(mu_angle, max_angle, angle_perturb, angle_1, angle_2)
        self.assertAlmostEqual(vmf_pdf(x1, mu, kappa), vmf_pdf(x2, mu, kappa), delta=0.01)

    def test_monte_carlo_integration(self):
        ITERATIONS = 10000
        MU, KAPPA  = angle_to_vector(RIGHT_ANGLE / 2), 0.5
        sum = 0
        for i in range(ITERATIONS):
            angle = np.random.uniform(0, 2*pi)
            vector = angle_to_vector(angle)
            sum += vmf_pdf(vector, MU, KAPPA)
        print('Approximate integral: ', pi * (sum / ITERATIONS))

    def test_plot_pdf(self):
        mu_angle = np.random.uniform(0, RIGHT_ANGLE)
        mu = angle_to_vector(mu_angle)
        kappa = 100
        theta = np.arange(mu_angle-pi, mu_angle+pi, 0.1)
        V = [angle_to_vector(t) for t in theta]
        y = [vmf_pdf(v, mu, kappa) for v in V]
        plt.plot(theta, y)
        plt.show()

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from von_mises_fisher.von_mises_fisher import vmf_pdf
from math import cos, sin, acos, asin, sqrt, pi

RIGHT_ANGLE = pi / 2

class PdfTest(unittest.TestCase):
    def test_2d_radial_symmetry(self):
        mu_angle = np.random.uniform(0, RIGHT_ANGLE)
        mu = np.array([cos(mu_angle), sin(mu_angle)])
        kappa = 0.5
        max_angle = min(acos(mu[0]), RIGHT_ANGLE - acos(mu[0]))
        angle_perturb = np.random.uniform(0, max_angle)
        angle_1 = max_angle + angle_perturb
        angle_2 = max_angle - angle_perturb
        x1 = np.array([cos(angle_1), sin(angle_1)])
        x2 = np.array([cos(angle_2), sin(angle_2)])
        print(mu_angle, max_angle, angle_perturb, angle_1, angle_2)
        print(vmf_pdf(x1, mu, kappa))
        print(vmf_pdf(x2, mu, kappa))

if __name__ == '__main__':
    unittest.main()
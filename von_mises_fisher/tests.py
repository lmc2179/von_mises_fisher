import unittest
import numpy as np
from von_mises_fisher.von_mises_fisher import vmf_pdf, vmf_mle, vmf_mixture_mle, vmf_log_pdf, fit_vmf_mixture_BIC
from math import cos, sin, acos, asin, sqrt, pi
from matplotlib import pyplot as plt
from scipy.integrate import quad

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
        self.assertAlmostEqual(vmf_pdf(x1, mu, kappa), vmf_pdf(x2, mu, kappa), delta=0.05) # There can be a high delta here due to numerical error

    def test_log_2d_radial_symmetry(self):
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
        self.assertAlmostEqual(vmf_log_pdf(x1, mu, kappa), vmf_log_pdf(x2, mu, kappa), delta=0.1) # There can be a high delta here due to numerical error

    def test_3d_radial_symmetry(self):
        kappa = 0.5
        mu = np.array([1, 0, 0])
        x1 = np.array([0, 1, 0])
        x2 = np.array([0, 0, 1])
        self.assertAlmostEqual(vmf_pdf(x1, mu, kappa), vmf_pdf(x2, mu, kappa), delta=0.01)
        self.assertAlmostEqual(vmf_pdf(-x1, mu, kappa), vmf_pdf(-x2, mu, kappa), delta=0.01)
        self.assertAlmostEqual(vmf_pdf(-x1, mu, kappa), vmf_pdf(x2, mu, kappa), delta=0.01)

    def test_3d_log_radial_symmetry(self):
        kappa = 0.5
        mu = np.array([1, 0, 0])
        x1 = np.array([0, 1, 0])
        x2 = np.array([0, 0, 1])
        self.assertAlmostEqual(vmf_log_pdf(x1, mu, kappa), vmf_log_pdf(x2, mu, kappa), delta=0.01)
        self.assertAlmostEqual(vmf_log_pdf(-x1, mu, kappa), vmf_log_pdf(-x2, mu, kappa), delta=0.01)
        self.assertAlmostEqual(vmf_log_pdf(-x1, mu, kappa), vmf_log_pdf(x2, mu, kappa), delta=0.01)

    def test_quadrature(self):
        MU, KAPPA  = angle_to_vector(RIGHT_ANGLE / 2), 0.5
        def vmf_pdf_angle(theta, mu, kappa):
            x = angle_to_vector(theta)
            return vmf_pdf(x, mu, kappa)
        print('Approximate integral: ', quad(vmf_pdf_angle, 0, 2*pi, args=(MU, KAPPA)))

    def test_mle(self):
        loose_pattern_data = np.array([angle_to_vector(0), angle_to_vector(-pi/4), angle_to_vector(pi/4)])
        tight_pattern_data = np.array([angle_to_vector(0), angle_to_vector(-pi/6), angle_to_vector(pi/6)])
        loose_mu, loose_kappa = vmf_mle(loose_pattern_data)
        tight_mu, tight_kappa = vmf_mle(tight_pattern_data)
        self.assertAlmostEqual(np.dot(loose_mu, tight_mu), 1.0)
        self.assertGreater(tight_kappa, loose_kappa)

    def test_mle_mixture(self):
        loose_pattern_data = np.array([angle_to_vector(0), angle_to_vector(-pi/4), angle_to_vector(pi/4)])
        tight_pattern_data = np.array([angle_to_vector(pi), angle_to_vector(pi-pi/4), angle_to_vector(pi+pi/4)])
        X = np.vstack((loose_pattern_data, tight_pattern_data))
        params = vmf_mixture_mle(X, 2, 100)
        print('Mixture parameters: ', params)

    def test_mle_mixture_BIC(self):
        loose_pattern_data = np.array([angle_to_vector(0), angle_to_vector(-pi/4), angle_to_vector(pi/4)])
        tight_pattern_data = np.array([angle_to_vector(pi), angle_to_vector(pi-pi/4), angle_to_vector(pi+pi/4)])
        sub_stacks = tuple([loose_pattern_data, tight_pattern_data]*10)
        X = np.vstack(sub_stacks)
        params, optimal_mixture_components = fit_vmf_mixture_BIC(X, [1,2], 100)
        print('BIC Mixture parameters: ', params)
        self.assertEqual(optimal_mixture_components, 2)

    @unittest.skip
    def test_plot_pdf(self):
        mu_angle = np.random.uniform(0, RIGHT_ANGLE)
        mu = angle_to_vector(mu_angle)
        kappa = 2
        theta = np.arange(mu_angle-pi, mu_angle+pi, 0.1)
        V = [angle_to_vector(t) for t in theta]
        y = [vmf_pdf(v, mu, kappa) for v in V]
        plt.plot(theta, y)
        plt.show()

if __name__ == '__main__':
    unittest.main()
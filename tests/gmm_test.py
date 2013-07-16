import unittest2 as unittest
import copy
import numpy as np
from gmm_specializer.gmm import GMM 

class BasicTests(unittest.TestCase):
    def test_init(self):
        gmm = GMM(3, 2, cvtype='diag')
        self.assertIsNotNone(gmm)

class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.D = 2
        self.N = 600
        self.M = 3
        np.random.seed(0)
        C = np.array([[0., -0.7], [3.5, .7]])
        C1 = np.array([[-0.4, 1.7], [0.3, .7]])
        Y = np.r_[
            np.dot(np.random.randn(self.N/3, 2), C1),
            np.dot(np.random.randn(self.N/3, 2), C),
            np.random.randn(self.N/3, 2) + np.array([3, 3]),
            ]
        self.X = Y.astype(np.float32)
    
    def test_pure_python(self):
        print "test pure python"
        gmm = GMM(self.M, self.D, cvtype='diag')
        means, covars = gmm.train_using_python(self.X)
        Y = gmm.predict_using_python(self.X)
        self.assertTrue(len(set(Y)) > 1)

    def test_training_once(self):
        print "test training once"
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        means0  = gmm0.components.means.flatten()
        covars0 = gmm0.components.covars.flatten()

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        means1  = gmm1.components.means.flatten()
        covars1 = gmm1.components.covars.flatten()

        self.assertAlmostEqual(likelihood0, likelihood1, places=3)
        for a,b in zip(means0, means1):   self.assertAlmostEqual(a,b)
        for a,b in zip(covars0, covars1): self.assertAlmostEqual(a,b)

    def test_prediction_once(self):
        print "test prediction once"
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        Y0 = gmm0.predict(self.X)

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        Y1 = gmm1.predict(self.X)

        for a,b in zip(Y0, Y1): self.assertAlmostEqual(a,b)
        self.assertTrue(len(set(Y0)) > 1)

    def test_training_repeat(self):
        print "test training repeat"
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        likelihood0 = gmm0.train(self.X)
        likelihood0 = gmm0.train(self.X)
        likelihood0 = gmm0.train(self.X)
        likelihood0 = gmm0.train(self.X)
        means0  = gmm0.components.means.flatten()
        covars0 = gmm0.components.covars.flatten()

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        likelihood1 = gmm1.train(self.X)
        likelihood1 = gmm1.train(self.X)
        likelihood1 = gmm1.train(self.X)
        likelihood1 = gmm1.train(self.X)
        means1  = gmm1.components.means.flatten()
        covars1 = gmm1.components.covars.flatten()

        self.assertAlmostEqual(likelihood0, likelihood1, places=3)
        for a,b in zip(means0, means1):   self.assertAlmostEqual(a,b)
        for a,b in zip(covars0, covars1): self.assertAlmostEqual(a,b)

    def test_prediction_full(self):
        print "test prediction full"
        gmm0 = GMM(self.M, self.D, cvtype='full')
        likelihood0 = gmm0.train(self.X)
        Y0 = gmm0.predict(self.X)

        gmm1 = GMM(self.M, self.D, cvtype='full')
        likelihood1 = gmm1.train(self.X)
        Y1 = gmm1.predict(self.X)

        for a,b in zip(Y0, Y1): self.assertAlmostEqual(a,b)
        self.assertTrue(len(set(Y0)) > 1)

    def test_getter_methods(self):
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        all_weights0  = gmm0.get_all_component_weights()
        all_means0 = gmm0.get_all_component_means()
        all_diag_covars0 = gmm0.get_all_component_diag_covariance()

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        all_weights1  = gmm1.get_all_component_weights()
        all_means1 = gmm1.get_all_component_means()
        all_diag_covars1 = gmm1.get_all_component_diag_covariance()

        self.assertAlmostEqual(likelihood0, likelihood1, places=3)

        for (a, b) in zip(all_weights0, all_weights1):
            self.assertAlmostEqual(a, b)

        for m in range(gmm0.M):
            for a, b in zip(all_means0[m], all_means1[m]):
                self.assertAlmostEqual(a, b)

        for a, b in zip(all_diag_covars0, all_diag_covars1):
            self.assertAlmostEqual(a, b)


if __name__ == '__main__':
    unittest.main()

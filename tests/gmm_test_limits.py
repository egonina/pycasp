import unittest2 as unittest
import numpy as np
from gmm_specializer.gmm import *

class SyntheticDataTests(unittest.TestCase):
    def read_data(self, in_file_name):
        feats = open(in_file_name, "r")

        out_feats = []        
        for line in feats:
            vals = line.split(" ")
            self.D = len(vals)
            for v in vals:
                out_feats.append(v)

        self.N = len(out_feats)/self.D

        return_points = np.array(out_feats, dtype=np.float32)
        return_points = return_points.reshape(self.N, self.D)

        return return_points

    def gen_data(self, N, D):
        self.D = D
        self.N = N
        feats = np.array(np.random.randn(N, D), dtype=np.float32)
        return feats

    def setUp(self):
        # read in training data
        #self.t1_data = self.read_data("gmm_sample_data/gmm_feats_n_500000.gmm")

        # generate training data
        self.t1_data = self.gen_data(1000000, 19)

    def test_training_and_classify_once(self):
        M = 512 
        gmm = GMM(M, self.D, cvtype='diag')
        gmm.train(self.t1_data)

        print gmm.get_all_component_weights()

if __name__ == '__main__':
    unittest.main()

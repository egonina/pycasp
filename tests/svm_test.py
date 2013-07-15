import unittest2 as unittest
import copy
import numpy as np
from svm_specializer.svm import * 

class BasicTests(unittest.TestCase):
    def test_init(self):
        svm = SVM()
        self.assertIsNotNone(svm)

class SyntheticDataTests(unittest.TestCase):
    def read_data(self, in_file_name):
        feats = open(in_file_name, "r")
        labels = []
        points = {}
        self.D = 0
        first_line = 1

        for line in feats:
            vals = line.split(" ")
            l = vals[0]
            labels.append(l)
            idx = 0
            for v in vals[1:]:
                if first_line:
                    self.D += 1
                f = v.split(":")[1].strip('\n')
                if idx not in points.keys():
                    points[idx] = []
                points[idx].append(f)
                idx += 1
            if first_line:
                first_line = 0

        self.N = len(labels)
        return_labels = np.array(labels, dtype=np.float32)
        points_list  = [] 

        for idx in points.keys():
           points_list.append(points[idx]) 

        return_points = np.array(points_list, dtype=np.float32)
        return_points = return_points.reshape(self.N, self.D)

        return return_labels, return_points

    def setUp(self):
        # read in training data
        self.t1_labels, self.t1_data = self.read_data("tests/svm_sample_data/svm_train_1.svm")
        self.t2_labels, self.t2_data = self.read_data("tests/svm_sample_data/svm_train_2.svm")

        # read in training data
        self.c_labels, self.c_data = self.read_data("tests/svm_sample_data/svm_classify.svm")

    def test_training_and_classify_once(self):
        svm = SVM()
        svm.train(self.t1_data, self.t1_labels, "linear")
        svm.classify(self.c_data, self.c_labels)

    def test_training_once(self):
        svm = SVM()
        a = svm.train(self.t2_data, self.t2_labels, "linear")

    def test_training_kernels(self):
        svm = SVM()
        a = svm.train(self.t1_data, self.t1_labels, "linear")
        a = svm.train(self.t2_data, self.t2_labels, "gaussian")

    def test_training_and_classify_twice(self):
        svm = SVM()
        svm.train(self.t1_data, self.t1_labels, "linear")
        svm.classify(self.c_data, self.c_labels)

        svm1 = SVM()
        svm1.train(self.t2_data, self.t2_labels, "linear")
        svm1.classify(self.c_data, self.c_labels)

if __name__ == '__main__':
    unittest.main()

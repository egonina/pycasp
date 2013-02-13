import unittest2 as unittest
import copy
import numpy as np
from gmm_specializer.gmm import GMM, compute_distance_BIC

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

        self.assertAlmostEqual(likelihood0, likelihood1)
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

        self.assertAlmostEqual(likelihood0, likelihood1)
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

class SpeechDataTests(unittest.TestCase):
    def setUp(self):
        self.X = np.ndfromtxt('./tests/speech_data.csv', delimiter=',', dtype=np.float32)
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.M = 5
        self.init_num_clusters = 16

    #def do_bic_agglomeration(self, gmm_list):
    #    print "test do bic agglomeration"
    #    # Get the events, divide them into an initial k clusters and train each GMM on a cluster
    #    per_cluster = self.N/self.init_num_clusters
    #    init_training = zip(gmm_list,np.vsplit(self.X, range(per_cluster, self.N, per_cluster)))
    #    for g, x in init_training:
    #        g.train(x)

    #    # Perform hierarchical agglomeration based on BIC scores
    #    best_BIC_score = 1.0
    #    while (best_BIC_score > 0 and len(gmm_list) > 1):
    #        num_clusters = len(gmm_list)
    #        # Resegment data based on likelihood scoring
    #        likelihoods = gmm_list[0].score(self.X)
    #        for g in gmm_list[1:]:
    #            likelihoods = np.column_stack((likelihoods, g.score(self.X)))
    #        most_likely = likelihoods.argmax(axis=1)
    #        # Across 2.5 secs of observations, vote on which cluster they should be associated with
    #        iter_training = {}
    #        for i in range(250, self.N, 250):
    #            votes = np.zeros(num_clusters)
    #            for j in range(i-250, i):
    #                votes[most_likely[j]] += 1
    #            iter_training.setdefault(gmm_list[votes.argmax()],[]).append(self.X[i-250:i,:])
    #        votes = np.zeros(num_clusters)
    #        for j in range((self.N/250)*250, self.N):
    #            votes[most_likely[j]] += 1
    #        iter_training.setdefault(gmm_list[votes.argmax()],[]).append(self.X[(self.N/250)*250:self.N,:])
    #        # Retrain the GMMs on the clusters for which they were voted most likely and
    #        # make a list of candidates for merging
    #        iter_bic_list = []
    #        for g, data_list in iter_training.iteritems():
    #            cluster_data =  data_list[0]
    #            for d in data_list[1:]:
    #                cluster_data = np.concatenate((cluster_data, d))
    #            cluster_data = np.ascontiguousarray(cluster_data)
    #            g.train(cluster_data)
    #            iter_bic_list.append((g,cluster_data))
    #
    #        # Keep any GMMs that lost all votes in candidate list for merging
    #        for g in gmm_list:
    #            if g not in iter_training.keys():
    #                iter_bic_list.append((g,None))            

    #        # Score all pairs of GMMs using BIC
    #        best_merged_gmm = None
    #        best_BIC_score = 0.0
    #        merged_tuple = None
    #        for gmm1idx in range(len(iter_bic_list)):
    #            for gmm2idx in range(gmm1idx+1, len(iter_bic_list)):
    #                g1, d1 = iter_bic_list[gmm1idx]
    #                g2, d2 = iter_bic_list[gmm2idx] 
    #                score = 0.0
    #                if d1 is not None or d2 is not None:
    #                    if d1 is not None and d2 is not None:
    #                        new_gmm, score = compute_distance_BIC(g1, g2, np.ascontiguousarray(np.concatenate((d1, d2))))
    #                    elif d1 is not None:
    #                        new_gmm, score = compute_distance_BIC(g1, g2, d1)
    #                    else:
    #                        new_gmm, score = compute_distance_BIC(g1, g2, d2)
    #                #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
    #                if score > best_BIC_score: 
    #                    best_merged_gmm = new_gmm
    #                    merged_tuple = (g1, g2)
    #                    best_BIC_score = score
    #        
    #        # Merge the winning candidate pair if its deriable to do so
    #        if best_BIC_score > 0.0:
    #            gmm_list.remove(merged_tuple[0]) 
    #            gmm_list.remove(merged_tuple[1]) 
    #            gmm_list.append(best_merged_gmm)

    #    return [ g.M for g in gmm_list] 

    def test_training_once(self):
        print "test speech training once"
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        means0  = gmm0.components.means.flatten()
        covars0 = gmm0.components.covars.flatten()

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        means1  = gmm1.components.means.flatten()
        covars1 = gmm1.components.covars.flatten()

        self.assertAlmostEqual(likelihood0, likelihood1)
        for a,b in zip(means0, means1):   self.assertAlmostEqual(a,b)
        for a,b in zip(covars0, covars1): self.assertAlmostEqual(a,b)
        
    def test_prediction_once(self):
        print "test speech prediction once"
        gmm0 = GMM(self.M, self.D, cvtype='diag')
        likelihood0 = gmm0.train(self.X)
        Y0 = gmm0.predict(self.X)

        gmm1 = GMM(self.M, self.D, cvtype='diag')
        likelihood1 = gmm1.train(self.X)
        Y1 = gmm1.predict(self.X)
        
        for a,b in zip(Y0, Y1): self.assertAlmostEqual(a,b)
        self.assertTrue(len(set(Y0)) > 1)

    ##TODO: Sometimes generates mysterious m-step cuda launch failures
    #def test_bic_agglomeration_diag(self):
    #    print "test big agglom diag"
    #    gmm_list = [GMM(self.M, self.D, cvtype='diag') for i in range(self.init_num_clusters)]
    #    ms = self.do_bic_agglomeration(gmm_list)
    #    self.assertItemsEqual(ms, [5, 10, 65])
    #    
#   #  def test_bic_agglomeration_full(self):
#   #      print "test big agglom full"
#   #      gmm_list = [GMM(self.M, self.D, cvtype='full') for i in range(self.init_num_clusters)]
#   #      ms = self.do_bic_agglomeration(gmm_list)
#   #      self.assertItemsEqual(ms, [5, 5, 5, 10, 15])

if __name__ == '__main__':
    unittest.main()

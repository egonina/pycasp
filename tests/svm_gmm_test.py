from gmm_specializer.gmm import *
from svm_specializer.svm import *

D = 2 
M = 3
N1 = 600
N2 = 1200
N3 = 1500

# "Training"
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
C1 = np.array([[-0.4, 1.7], [0.3, .7]])
Y1 = np.r_[
    np.dot(np.random.randn(N1/3, 2), C1),
    np.dot(np.random.randn(N1/3, 2), C),
    np.random.randn(N1/3, 2) + np.array([3, 3]),
    ]
X1 = Y1.astype(np.float32)

np.random.seed(0)
C = np.array([[0.1, 7.2], [-3.6, 9.7]])
C1 = np.array([[0.4, 5.7], [-0.3, 8.7]])
Y2 = np.r_[
    np.dot(np.random.randn(N2/3, 2), C1),
    np.dot(np.random.randn(N2/3, 2), C),
    np.random.randn(N2/3, 2) + np.array([3, 3]),
    ]
X2 = Y2.astype(np.float32)
    
gmm0 = GMM(M, D, cvtype='diag')
likelihood0 = gmm0.train(X1)

gmm1 = GMM(M, D, cvtype='diag')
likelihood1 = gmm1.train(X2)

all_means = np.hstack((gmm0.components.means.reshape(1, M*D),\
                       gmm1.components.means.reshape(1,M*D))).\
                       reshape(2, M*D, order='F')
labels = np.array([1.0, -1.0], dtype=np.float32)

svm = SVM()
svm.train(all_means, labels, "linear")

# "Testing"
np.random.seed(0)
C = np.array([[0.4, 9.7], [0.5, -4.1]])
C1 = np.array([[0.14, 5.5], [3.3, 3.7]])
Y3 = np.r_[
    np.dot(np.random.randn(N3/3, 2), C1),
    np.dot(np.random.randn(N3/3, 2), C),
    np.random.randn(N3/3, 2) + np.array([3, 3]),
    ]
X3 = Y3.astype(np.float32)

gmm3 = GMM(M, D, cvtype='diag')
likelihood3 = gmm3.train(X2)
means3 = gmm3.components.means
means3 = means3.reshape(1,M*D)

c_labels = np.array([1.0], dtype=np.float32)

svm.classify(means3, c_labels)

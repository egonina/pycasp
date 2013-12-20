from gmm_specializer.gmm import *
import MySQLdb as mdb
import pickle
import time
import binascii
import array
import sqlite3
import msdtools

ubm_t_feats_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_features_1M_008.pkl"
ubm_r_feats_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_rhythm_features_1M_008.pkl"
ubm_t_params_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_timbre_params.pkl"
ubm_r_params_pkl = "/disk1/home_user/egonina/msd_database/pickles/ubm_rhythm_params.pkl"

#=====================================
#          UBM ADAPTATION 
#=====================================
def adapt_means(ubm_means, ubm_covars, ubm_weights, new_means, new_weights, T):
    n_i = new_weights*T
    alpha_i = n_i/(n_i+10)
    new_means[np.isnan(new_means)] = 0.0
    return_means = (alpha_i*new_means.T+(1-alpha_i)*ubm_means.T).T
    diag_covars = np.diagonal(ubm_covars, axis1=1, axis2=2)
    
    return_means = (np.sqrt(ubm_weights)*(1/np.sqrt(diag_covars.T))*return_means.T).T
    return return_means

def adapt_model(feats, ubm_params, M):
    # train GMM on features
    D = feats.shape[1]
    updated_means = np.array(ubm_params['means'], dtype=np.float32)

    for it in range(1): # adaptation loop
        gmm = GMM(M, D, means=updated_means, covars=np.array(ubm_params['covars']), \
              weights=np.array(ubm_params['weights']), cvtype='diag')
        gmm.train(feats, max_em_iters=1)
    
        new_means = gmm.components.means
        new_weights = gmm.components.weights
        T = feats.shape[0]
        updated_means = adapt_means(ubm_params['means'], \
                        ubm_params['covars'], ubm_params['weights'], \
                        new_means, new_weights, T).flatten('C')

    return updated_means

#=====================================
#           UBM TRAINING 
#=====================================
def get_UBM_features():
    '''
    gets the features for timbre and rhythm ubm training 
    from pickle file 
    '''
    p = open(ubm_t_feats_pkl, "rb")
    ubm_timbre_features = np.array(pickle.load(p), dtype=np.float32)
    p.close()
    p = open(ubm_r_feats_pkl, "rb")
    ubm_rhythm_features = np.array(pickle.load(p), dtype=np.float32)
    p.close()

    return ubm_timbre_features, ubm_rhythm_features 

def train_and_pickle_UBM(M):
    '''
    train the UBM on a subset of features
    for now, get features from pickle file
    '''
    ubm_timbre_features, ubm_rhythm_features  = get_UBM_features()

    # Train Timbre UBM
    D = ubm_timbre_features.shape[1]
    num_timbre_feats = ubm_timbre_features.shape[0]
    print "--- total number of timbre ubm features:\t", num_timbre_feats, " -----"
    timbre_ubm = GMM(M,D,cvtype='diag')

    train_st = time.time()
    timbre_ubm.train(ubm_timbre_features)
    train_total = time.time() - train_st
    print "--- timbre ubm training time:\t", train_total, " -----"

    # Train Rhythm UBM
    D = ubm_rhythm_features.shape[1]
    num_rhythm_feats = ubm_rhythm_features.shape[0]
    print "--- total number of rhythm ubm features:\t", num_rhythm_feats, " -----"
    rhythm_ubm = GMM(M,D,cvtype='diag')

    train_st = time.time()
    rhythm_ubm.train(ubm_rhythm_features, max_em_iters=5)
    train_total = time.time() - train_st
    print "--- rhythm ubm training time:\t", train_total, " -----"

    #make dict of ubm parameters
    timbre_ubm_params = {}
    timbre_ubm_params['means'] = timbre_ubm.components.means
    timbre_ubm_params['covars'] = timbre_ubm.components.covars
    timbre_ubm_params['weights'] = timbre_ubm.components.weights

    rhythm_ubm_params = {}
    rhythm_ubm_params['means'] = rhythm_ubm.components.means
    rhythm_ubm_params['covars'] = rhythm_ubm.components.covars
    rhythm_ubm_params['weights'] = rhythm_ubm.components.weights

    #pickle the parameters
    p = open(ubm_t_params_pkl, "wb")
    pickle.dump(timbre_ubm_params, p, True)
    p.close()

    p = open(ubm_r_params_pkl, "wb")
    pickle.dump(rhythm_ubm_params, p, True)
    p.close()

    return timbre_ubm_params, rhythm_ubm_params

def get_UBM_parameters(M, from_pickle=False):
    if from_pickle:
        p = open(ubm_t_params_pkl, "rb")
        timbre_ubm_params = pickle.load(p)
        p.close()
        p = open(ubm_r_params_pkl, "rb")
        rhythm_ubm_params = pickle.load(p)
        p.close()
    else:
        timbre_ubm_params, rhythm_ubm_params = train_and_pickle_UBM(M)

    return timbre_ubm_params, rhythm_ubm_params

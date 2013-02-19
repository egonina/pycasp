import os
import sys
import math
import timeit
import copy
import time
import struct
import pickle
import scipy.stats.mstats as stats
import numpy as np
from gmm_specializer.gmm import GMM 
from svm_specializer.svm import SVM 

def read_features(f_file_name, sp_file_name):
    print f_file_name, sp_file_name
    f = open(f_file_name, "rb")

    print "...Reading in HTK feature file..."
    
    #=== Read Feature File ==
    try:
        nSamples = struct.unpack('>i', f.read(4))[0]
        sampPeriod = struct.unpack('>i', f.read(4))[0]
        sampSize = struct.unpack('>h', f.read(2))[0]
        sampKind = struct.unpack('>h', f.read(2))[0]

        print "INFO: total number of frames read: ", nSamples
        total_num_frames = nSamples
            
        D = sampSize/4 #dimension of feature vector
        l = []
        count = 0
        while count < (nSamples * D):
            bFloat = f.read(4)
            fl = struct.unpack('>f', bFloat)[0]
            l.append(fl)
            count = count + 1
    finally:
        f.close()

    #=== Prune to Speech Only ==
    print "...Reading in speech/nonspeech file..."
    pruned_list = []
    num_speech_frames = nSamples            

    if sp_file_name:
        sp = open(sp_file_name, "r")
                    
        l_start = []
        l_end = []
        num_speech_frames = 0
        for line in sp:
            s = line.split(' ')
            st = math.floor(100 * float(s[2]) + 0.5)
            en = math.floor(100 * float(s[3].replace('\n','')) + 0.5)
            st1 = int(st)
            en1 = int(en)
            l_start.append(st1*19)
            l_end.append(en1*19)
            num_speech_frames = num_speech_frames + (en1 - st1 + 1)

        print "INFO: total number of speech frames: ", num_speech_frames

        total = 0
        for start in l_start:
            end = l_end[l_start.index(start)]
            total += (end/19 - start/19 + 1)
            x = 0
            index = start
            while x < (end-start+19):
                pruned_list.append(l[index])
                index += 1
                x += 1
    else: #no speech file, take in all features
        pruned_list = l

    floatArray = np.array(pruned_list, dtype = np.float32)

    return num_speech_frames, floatArray 
    

def adapt_means(ubm_means, ubm_covars, ubm_weights, new_means, new_weights, T):
    n_i = new_weights*T
    alpha_i = n_i/(n_i+10)
    new_means[np.isnan(new_means)] = 0.0
    return_means = (alpha_i*new_means.T+(1-alpha_i)*ubm_means.T).T
    diag_covars = np.diagonal(ubm_covars, axis1=1, axis2=2)
    
    return_means = (np.sqrt(ubm_weights)*(1/np.sqrt(diag_covars.T))*return_means.T).T
    return return_means

def get_htk_features(directory, from_pickle=False):
    if from_pickle:
        feats = pickle.load(open("htk_feats_"+directory+".pkl", "rb"))
        all_feats = feats['all_feats']
        total_num_feats = feats['total_num_feats']
        speaker1_feats = feats['speaker1_feats']
        speaker2_feats = feats['speaker2_feats'] 
    else:
        # === GET UBM FEATS ===
        speaker1_htk_dir = "/disk1/home_user/egonina/speaker_id/" + directory + "/speaker1/htk/"
        speaker1_sph_dir = "/disk1/home_user/egonina/speaker_id/" + directory + "/speaker1/spch/"
        speaker2_htk_dir = "/disk1/home_user/egonina/speaker_id/" + directory + "/speaker2/htk/"
        speaker2_sph_dir = "/disk1/home_user/egonina/speaker_id/" + directory + "/speaker2/spch/"
        all_feats = []
        speaker1_feats = []
        speaker2_feats = []
        total_num_feats = 0

        for htk_file in os.listdir(speaker1_htk_dir):
            htk_file_path = speaker1_htk_dir + htk_file
            sph_file_path = speaker1_sph_dir + htk_file.split(".")[0] + ".spch"
            num_feats, feats = read_features(htk_file_path, sph_file_path)
            all_feats.append(feats)
            speaker1_feats.append(feats)
            total_num_feats += num_feats

        for htk_file in os.listdir(speaker2_htk_dir):
            htk_file_path = speaker2_htk_dir + htk_file
            sph_file_path = speaker2_sph_dir + htk_file.split(".")[0] + ".spch"
            num_feats, feats = read_features(htk_file_path, sph_file_path)
            all_feats.append(feats)
            speaker2_feats.append(feats)
            total_num_feats += num_feats

        feats = {}
        feats['all_feats'] = all_feats
        feats['total_num_feats'] = total_num_feats
        feats['speaker1_feats'] = speaker1_feats
        feats['speaker2_feats'] = speaker2_feats

        pickle.dump(feats, open("htk_feats_"+directory+".pkl", "wb"))

    return all_feats, total_num_feats, speaker1_feats, speaker2_feats

def train_UBM(M, D, all_feats, total_num_feats):

    ubm_feats =  all_feats[0]

    for d in all_feats[1:]:
        ubm_feats = np.concatenate((ubm_feats, d))

    print "total feats: ", total_num_feats
    ubm_feats = ubm_feats.reshape(total_num_feats, D)

    # === TRAIN UBM ===
    ubm = GMM(M, D, cvtype='diag')
    ubm.train(ubm_feats, max_em_iters=5)

    ubm_params = {}
    ubm_params['means'] = ubm.components.means
    ubm_params['covars'] = ubm.components.covars
    ubm_params['weights'] = ubm.components.weights

    return ubm_params

def adapt_UBM(ubm_params, data):
    updated_means = np.array(ubm_params['means'], dtype=np.float32)

    for it in range(1): # adaptation loop
        gmm = GMM(M, D, means=updated_means, covars=np.array(ubm_params['covars']),\
                  weights=np.array(ubm_params['weights']), cvtype='diag')
        gmm.train(data, max_em_iters=1)
    
        new_means = gmm.components.means
        new_weights = gmm.components.weights
        T = data.shape[0]
        updated_means = adapt_means(ubm_params['means'], ubm_params['covars'],\
                                    ubm_params['weights'], new_means, new_weights, T).flatten('C')

    return updated_means

def adapt_UBM_to_two_speakers(speaker1_feats, speaker2_feats, ubm_params):

    # === ADAPT UBM, CONSTRUCT SVM TRAINIG FEATURES ===
    speaker1_svm_feats = []
    speaker2_svm_feats = []

    for data in speaker1_feats:
        data = data.reshape(data.shape[0]/D, D)
        updated_means = adapt_UBM(ubm_params, data)
        speaker1_svm_feats.append(updated_means) 

    for data in speaker1_feats:
        data = data.reshape(data.shape[0]/D, D)
        updated_means = adapt_UBM(ubm_params, data)
        speaker2_svm_feats.append(updated_means) 

    return speaker1_svm_feats, speaker2_svm_feats

def train_SVM(svm_train_feats, svm_labels):
    # === TRAIN SVM ===
    svm = SVM()
    svm.train(svm_train_feats, svm_labels, "gaussian") 
    return svm

def concat_svm_features(speaker1_svm_feats, speaker2_svm_feats):
    # TODO: figure out a less hacky way to do this... :(
    # stack the svm training features
    svm_train_feats = np.hstack((speaker1_svm_feats[0].reshape(1, M*D),\
                                 speaker1_svm_feats[1].reshape(1, M*D),\
                                 speaker1_svm_feats[2].reshape(1, M*D),\
                                 speaker1_svm_feats[3].reshape(1, M*D),\
                                 speaker1_svm_feats[4].reshape(1, M*D),\
                                 speaker2_svm_feats[0].reshape(1, M*D),\
                                 speaker2_svm_feats[1].reshape(1, M*D),\
                                 speaker2_svm_feats[2].reshape(1, M*D),\
                                 speaker2_svm_feats[3].reshape(1, M*D),\
                                 speaker2_svm_feats[4].reshape(1, M*D))).reshape(10, M*D, order='F')
    return svm_train_feats

if __name__ == '__main__':
    
    M = 64
    D = 19

    # ============================ TRAINING PHASE =========================

    # --- Train & Adapt UBM ---
    all_train_feats, total_num_feats,\
    sp1_htk_train_feats, sp2_htk_train_feats = get_htk_features("train", from_pickle=True)

    ubm_params = train_UBM(M, D, all_train_feats, total_num_feats)
    sp1_svm_train_feats, sp2_svm_train_feats = adapt_UBM_to_two_speakers(sp1_htk_train_feats,\
                                                                         sp2_htk_train_feats,\
                                                                         ubm_params) 

    # --- Train SVM ---
    svm_train_feats = concat_svm_features(sp1_svm_train_feats, sp2_svm_train_feats)
    svm_train_labels = np.ones(10, dtype=np.float32)
    svm_train_labels[5:] *= -1
    speaker_svm = train_SVM(svm_train_feats, svm_train_labels)

    # ============================ TESTING PHASE =========================

    all_test_feats, total_num_test_feats,\
    sp1_htk_test_feats, sp2_htk_test_feats = get_htk_features("test", from_pickle=True)

    for htk_feats in sp1_htk_test_feats:
        htk_feats = htk_feats.reshape(htk_feats.shape[0]/D, D)
        svm_feats = adapt_UBM(ubm_params, htk_feats)
        svm_feats = svm_feats.reshape(1, M*D)
        lab = np.ones(1)
        result = speaker_svm.classify(svm_feats, lab)

        
    for htk_feats in sp2_htk_test_feats:
        htk_feats = htk_feats.reshape(htk_feats.shape[0]/D, D)
        svm_feats = adapt_UBM(ubm_params, htk_feats)
        svm_feats = svm_feats.reshape(1, M*D)
        lab = np.ones(1) * -1.0
        result = speaker_svm.classify(svm_feats, lab)

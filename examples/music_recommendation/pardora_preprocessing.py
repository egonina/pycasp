from gmm_specializer.gmm import *
import MySQLdb as mdb
import pickle
import time
import binascii
import array
import sqlite3
import msdtools
import unicodedata
import collab

#pardora imports
import pardora_db
import pardora_ubm

song_id_pkl = "/disk1/home_user/egonina/msd_database/pickles/song_ids_1M.pkl"
CHUNK_SIZE = 50 
NORM_CHUNK_SIZE = 1000

#=====================================
#         HELPERS 
#=====================================
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]
    
#=====================================
#         RHYTHM COMPUTATION 
#=====================================

def compute_and_add_rhythm_feats(db_cursor):
    print "............... Computing and Adding Rhythm Features To DB ...................."
    p = open(song_id_pkl, "rb")    
    song_ids = pickle.load(p)
    p.close()

    song_id_chunks = chunks(song_ids, CHUNK_SIZE)
    chunk_count = 0

    # get all song_ids
    sql_query = "SELECT song_id FROM songs1m_rhythm;"
    db_cursor.execute(sql_query)
    all_songs = db_cursor.fetchall()
    all_songs_list = []
    for s in all_songs:
        s = s[0]
        all_songs_list.append(s)

    total_time = time.time()
    for chunk in song_id_chunks:
        if chunk_count > 2908:
            print "==== CHUNK: ", chunk_count, " ===="
            st = time.time()

            songs = pardora_db.get_timbre_features_for_song_ids(chunk, db_cursor)
            
            for s in songs:
                t_feats =  np.ndarray((s[2],s[3]), buffer=s[0])
                segments =  np.ndarray((s[4],), buffer=s[1])
                song_id = s[5]

                #compute rhythm features
                onset_coefs, onset_pattern = msdtools.rhythm_features(t_feats, segments)

                if onset_coefs is not None:
                    if song_id not in all_songs_list:
                        pardora_db.add_rhythm_feats_to_db(conn, c, song_id, onset_coefs, db_cursor)
                        all_songs_list.append(song_id)

            print "INFO: chunk rhythm compute time:", time.time() - st
            conn.commit()

        chunk_count += 1 

    print "================================"
    print "INFO: TOTAL TIME FOR RHYTHM COMP TIME: ", time.time() - total_time 
    print "================================"

    return

#=====================================
#       SUPERVECTOR COMPUTATION 
#=====================================

def compute_and_add_song_svs(timbre_ubm_params, rhythm_ubm_params, db_cursor):
    print "............... Computing and Adding Supervectors To DB ...................."
    p = open(song_id_pkl, "rb")    
    song_ids = pickle.load(p)
    p.close()

    song_id_chunks = chunks(song_ids, CHUNK_SIZE)
    chunk_count = 0

    t_mean_to_use = np.zeros(1)
    t_sv = np.zeros(1)
    r_mean_to_use = np.zeros(1)
    r_sv = np.zeros(1)

    all_songs_list = []

    total_time = time.time()
    for chunk in song_id_chunks:
          song_id_list = []
          timbre_sv_arr = []
          rhythm_sv_arr = []
          print "==== CHUNK: ", chunk_count, "===="
          chunk_count+=1

          chunk_time = time.time()

          songs = pardora_db.get_rhythm_features_for_song_ids(chunk, db_cursor)
          
          for s in songs:
              if s[1] is not None and s[2] is not None:
                  feats =  np.array(np.ndarray((s[1],s[2]), buffer=s[0]), dtype=np.float32)
                  feats_t = feats.T
                  rhythm_sv = pardora_ubm.adapt_model(feats_t, rhythm_ubm_params, M)
                  rhythm_sv_arr.append(rhythm_sv)
                  song_id_list.append(s[3])

          print "INFO: Rhythm SV comp time: ", time.time() - st

          st = time.time()
          songs = pardora_db.get_timbre_features_for_song_ids(chunk, db_cursor)

          for s in songs:
              if s[3] in song_id_list:
                  feats =  np.array(np.ndarray((s[1],s[2]), buffer=s[0]), dtype=np.float32)
                  timbre_sv = pardora_ubm.adapt_model(feats, timbre_ubm_params, M)
                  timbre_sv_arr.append(timbre_sv)

          print "INFO: Timbre SV comp time: ", time.time() - st
          
          st = time.time()
          t_sv = np.vstack(timbre_sv_arr)
          del timbre_sv_arr
          r_sv = np.vstack(rhythm_sv_arr)
          del rhythm_sv_arr

          if chunk_count == 1:
              t_mean_to_use = np.mean(t_sv, axis=0)
              r_mean_to_use = np.mean(r_sv, axis=0)

          t_sv = msdtools.mcs_norm(t_sv.T, t_mean_to_use).T
          r_sv = msdtools.mcs_norm(r_sv.T, r_mean_to_use).T
          
          print "INFO: MCS norm computation time: ", time.time() - st

          st = time.time()
          p_means_t = np.zeros(len(song_id_list))
          p_sigmas_t = p_means_t.copy()
          p_means_t = p_means_t.copy() 
          p_sigmas_t = p_means_t.copy()

          st = time.time()

          p_means_t, p_sigmas_t = msdtools.p_norm_params_chunk(t_sv.T, t_sv.T, NORM_CHUNK_SIZE)
          p_means_r, p_sigmas_r = msdtools.p_norm_params_chunk(r_sv.T, r_sv.T, NORM_CHUNK_SIZE)

          print "INFO: P-means computation time: ", time.time() - st


          st = time.time()
          idx = 0
          for s_id in song_id_list:
              t = np.array(t_sv[idx])
              r = np.array(r_sv[idx])
              if s_id not in all_songs_list:
                  pardora_db.add_sv_and_p_vals_to_db(conn, c, s_id, t, r, \
                                                     p_means_t[idx], p_sigmas_t[idx],\
                                                     p_means_r[idx], p_sigmas_r[idx],\
                                                     db_cursor)
                  all_songs_list.append(s_id)
              idx += 1

          conn.commit()

          print "INFO: Database update time: ", time.time() - st

          print "INFO: TOTAL CHUNK TIME:", time.time() - chunk_time

    print "=============================================="
    print "INFO: TOTAL TIME FOR SV COMP TIME: ", time.time() - total_time
    print "=============================================="
    d = {}   
    d['t_sv_mean'] = t_mean_to_use
    d['t_sv_sample'] = t_sv
    d['r_sv_mean'] = r_mean_to_use
    d['r_sv_sample'] = r_sv

    p = open(norm_param_pkl, "wb")
    pickle.dump(d, p, True)
    p.close()

    return

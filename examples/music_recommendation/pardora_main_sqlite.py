from gmm_specializer.gmm import *
import numpy as np
import MySQLdb as mdb
import pickle
import time
import binascii
import array
import sqlite3
import msdtools
import unicodedata
import collab
import json
import sqlite3 as lite

# Pardora-specific imports
import pardora_db_sqlite
import pardora_ubm
import pardora_preprocessing

norm_param_pkl = "/disk1/home_user/egonina/msd_database/pickles/norm_param_pkl_1M.pkl"
l1_output_pkl = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l1.pkl"
l2_output_pkl = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l2.pkl"
l3_output_pkl = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l3.pkl"
l4_output_pkl = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l4.pkl"

song_list_10k = "/disk1/home_user/egonina/msd_database/10k_songs.json"
l1_output_json = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l1.json"
l2_output_json = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l2.json"
l3_output_json = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l3.json"
l4_output_json = "/disk1/home_user/egonina/pardora_demo/sample_output/lady_gaga/pardora_l4.json"

database = "/disk1/home_user/egonina/msd_database/msd_subset.db"

CF_NEIGHBORS = 100 
M = 64

class Pardora:
    #=================================================
    #        WRAPPERS TO PREPROCESSING FUNCTIONS  
    #=================================================
    def create_rhythm_table(self):
        pardora_db_sqlite.create_rhythm_table(self.conn, self.cursor)
        return

    def drop_rhythm_table(self):
        pardora_db_sqlite.drop_rhythm_table(self.conn, self.cursor)
        return

    def compute_and_add_rhythm_feats(self):
        pardora_preprocessing.compute_and_add_rhythm_feats(self.cursor)
        return

    def create_sv_table(self):
        pardora_db_sqlite.create_sv_table(self.conn, self.cursor)
        return

    def drop_sv_table(self):
        pardora_db_sqlite.drop_sv_table(self.conn, self.cursor)
        return

    def compute_and_add_song_svs(self, timbre_ubm_params, rhythm_ubm_params):
        pardora_preprocessing.compute_and_add_song_svs(timbre_ubm_params, rhythm_ubm_params, self.cursor)
        return

    def get_all_song_data(self):
        song_list = pardora_db_sqlite.get_all_song_data(self.cursor)
        return song_list
    
    #=====================================
    #         QUERY COMPUTATIONS 
    #=====================================
    def get_query_data(self, song_id_list):
        p = open(norm_param_pkl, "rb")
        song_sv_dict = pickle.load(p)
        p.close()

        print "NUMBER OF SONGS IN QUERY: ", len(song_id_list)
        if len(song_id_list) > 1:
            timbre_result, rhythm_result = pardora_db_sqlite.get_song_features_from_query(song_id_list, self.cursor)

            t_feature_list = []
            r_feature_list = []
    
            for row in timbre_result:
               feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
               t_feature_list.append(feats)

            timbre_features = np.array(np.concatenate(t_feature_list))

            for row in rhythm_result:
               feats =  np.array(np.ndarray((row[0],row[1]), buffer=row[2]), dtype=np.float32)
               feats = feats.T
               r_feature_list.append(feats)

            rhythm_features = np.array(np.concatenate(r_feature_list))


            print "INFO: Timbre features shape:", timbre_features.shape
            print "INFO: Rhythm features shape:", rhythm_features.shape

            query_timbre_sv = pardora_ubm.adapt_model(timbre_features, self.timbre_ubm_params, M)
            query_rhythm_sv = pardora_ubm.adapt_model(rhythm_features, self.rhythm_ubm_params, M)

            query_timbre_sv = msdtools.mcs_norm(query_timbre_sv, song_sv_dict['t_sv_mean'])
            query_rhythm_sv = msdtools.mcs_norm(query_rhythm_sv, song_sv_dict['r_sv_mean'])

            p_mean_t, p_sigma_t = msdtools.p_norm_params_single(query_timbre_sv, song_sv_dict['t_sv_sample'].T)
            p_mean_r, p_sigma_r = msdtools.p_norm_params_single(query_rhythm_sv, song_sv_dict['r_sv_sample'].T)

            query_dict = {}
            query_dict['q_t_sv'] = query_timbre_sv
            query_dict['q_r_sv'] = query_rhythm_sv
            query_dict['p_mean_t'] = p_mean_t
            query_dict['p_mean_r'] = p_mean_r
            query_dict['p_sigma_t'] = p_sigma_t
            query_dict['p_sigma_r'] = p_sigma_r
        else:
            query_dict = pardora_db_sqlite.get_song_sv_data(song_id_list[0], self.cursor)

        return query_dict 

    def get_query_data_multi_query(self, song_id_list):
        p = open(norm_param_pkl, "rb")
        song_sv_dict = pickle.load(p)
        p.close()

        query_dicts = pardora_db_sqlite.get_song_svs_multi_query(song_id_list, self.cursor)

        return query_dicts

    def get_collab_info(self, song_id_list):
        output_song_ids, output_similarity = \
                collab.filter_compound_query(song_id_list, num_neighbors = CF_NEIGHBORS)

        collab_song_data = {}
        if output_song_ids is not None:
            idx = 0
            for s in output_song_ids:
                if s not in song_id_list:
                    collab_song_data[s] = output_similarity[idx]
                idx += 1

        return collab_song_data 

    def get_collab_info_multi_query(self, song_id_list):
        output_song_ids, output_similarity = \
                collab.filter_multiple_queries(song_id_list, num_neighbors = CF_NEIGHBORS)
        
        # collab_song_data[input_song_id] -> dictionary neighbor_song_id -> cf_score
        collab_song_data = {}
        for idx in range(len(song_id_list)):
            song_id = song_id_list[idx]
            cf_nn = output_song_ids[idx] #should be a list..
            cf_scores = output_similarity[idx] #should be a list..
            collab_song_data[song_id] = {}
            for nn_idx in range(len(cf_nn)):
                if song_id != cf_nn[nn_idx]:
                    collab_song_data[song_id][cf_nn[nn_idx]] = cf_scores[nn_idx]

        return collab_song_data 


    def get_nn_dict(self, qd, NN, fanout, parent_cf_score=0.0):
        song_ids = []
        title_artist = []
        mta = []
        t_supervectors = []
        t_p_means = []
        t_p_sigmas = []
        r_supervectors = []
        r_p_means = []
        r_p_sigmas = []
        cf_distances = []
        for song in NN.keys():
            song_ids.append(song)        
            t_supervectors.append(NN[song]['t_sv'])
            r_supervectors.append(NN[song]['r_sv'])
            t_p_means.append(NN[song]['p_mean_t'])
            r_p_means.append(NN[song]['p_mean_r'])
            t_p_sigmas.append(NN[song]['p_sigma_t'])
            r_p_sigmas.append(NN[song]['p_sigma_r'])
            title = NN[song]['title']
            artist = NN[song]['artist_name']
            mode = NN[song]['mode']
            tempo = NN[song]['tempo']
            artist_hottness = NN[song]['artist_hottness']
            cf_distances.append(NN[song]['cf_score'])
            title_artist.append((title, artist))
            mta.append((mode, tempo, artist_hottness))
            
        all_t_sv = np.vstack((t_supervectors))
        all_t_p_means = np.array(np.hstack((t_p_means)), dtype=np.float32)
        all_t_p_sigmas = np.array(np.hstack((t_p_sigmas)), dtype=np.float32)
        all_r_sv = np.vstack((r_supervectors))
        all_r_p_means = np.array(np.hstack((r_p_means)), dtype=np.float32)
        all_r_p_sigmas = np.array(np.hstack((r_p_sigmas)), dtype=np.float32)
        
        timbre_dist = msdtools.p_norm_distance_single(qd['q_t_sv'], all_t_sv.T, qd['p_mean_t'], all_t_p_means, qd['p_sigma_t'], all_t_p_sigmas)
        rhythm_dist = msdtools.p_norm_distance_single(qd['q_r_sv'], all_r_sv.T, qd['p_mean_r'], all_r_p_means, qd['p_sigma_r'], all_r_p_sigmas)

        cf_dist = np.array(cf_distances, dtype=np.float32)

        total_dist = 0.7*timbre_dist + 0.3*rhythm_dist + cf_dist+ parent_cf_score 
        sorted_indices = np.argsort(total_dist)
        sorted_distances = np.sort(total_dist)

        close_songs = {} 
        count = 0
        for index in sorted_indices[:fanout]:
            song_id = song_ids[index]
            close_songs[song_id] = {}
            close_songs[song_id]['class'] = "Node" 
            close_songs[song_id]['song_id'] = song_id 
            close_songs[song_id]['artist_name'] = title_artist[index][1]
            close_songs[song_id]['title'] = title_artist[index][0]
            close_songs[song_id]['dist_to_parent'] = float(sorted_distances[count])
            close_songs[song_id]['mode'] = mta[index][0] 
            close_songs[song_id]['tempo'] = mta[index][1] 
            close_songs[song_id]['artist_hottness'] = mta[index][2] 
            count += 1

        return close_songs

    def get_nn_one_query(self, song_id_list, fanout):
        print "*********************************************************"
        print "              GET NN COMPOUND QUERY                      "
        print "*********************************************************"
        st = time.time()
        query_dict = self.get_query_data(song_id_list)
        print "INFO: Step 1, Get query data and supervectors:", time.time() - st

        st = time.time()
        collab_song_info = self.get_collab_info(song_id_list)
        print "INFO: Step 2, Get collaborative filtering results:", time.time() - st

        st = time.time()
        cf_data_query_time = 0
        dist_comp_time = 0

        nn_cf_scores = [] 

        if len(collab_song_info.keys()) > 0:
            t1 = time.time()
            close_songs_dict = pardora_db_sqlite.get_cf_songs_data(collab_song_info, self.cursor)
            cf_data_query_time += time.time() - t1 

            t2 = time.time()
            nn_dict = self.get_nn_dict(query_dict, close_songs_dict, fanout)
            dist_comp_time += time.time() - t2

            # keep track of cf score of the neighbors separately
            for n in nn_dict.keys():
                nn_cf_scores.append((n, collab_song_info[n]))

        else:
            print "No collaborative filtering neighbors found."
            nn_dict = None 


        print "INFO: Step 3, Compute closest songs:", time.time() - st,\
              ";\n \tCF data gather (", cf_data_query_time, "), Dist comp (", dist_comp_time, ")"
        return nn_dict, nn_cf_scores 

    def get_nn_multi_query(self, song_id_list, nn_cf_scores, fanout):
        print "*********************************************************"
        print "              GET NN MULTI QUERY                      "
        print "*********************************************************"
        st = time.time()
        query_dicts = self.get_query_data_multi_query(song_id_list)
        print "INFO: Step 1, Get query data and supervectors:", time.time() - st

        st = time.time()
        collab_song_infos = self.get_collab_info_multi_query(song_id_list)
        print "INFO: Step 2, Get collaborative filtering results:", time.time() - st

        total_nn_dict = {}
        total_id_list = []
        out_nn_cf_scores = []
        st = time.time()
        cf_data_query_time = 0
        dist_comp_time = 0

        for input_song in nn_cf_scores:
            input_song_id = input_song[0]
            input_song_cf_score = input_song[1]

            if len(collab_song_infos[input_song_id].keys()) > 0:
                t1 = time.time()
                close_songs_dict = pardora_db_sqlite.get_cf_songs_data(collab_song_infos[input_song_id], self.cursor)
                cf_data_query_time += time.time() - t1 

                t2 = time.time()
                nn_dict = self.get_nn_dict(query_dicts[input_song_id],\
                                           close_songs_dict, fanout,\
                                           parent_cf_score=input_song_cf_score)
                dist_comp_time += time.time() - t2 

            else:
                print "No collaborative filtering neighbors found."
                nn_dict = None 

            total_nn_dict[input_song_id] = nn_dict
            for k in nn_dict.keys(): 
                total_id_list.append(k)
                out_nn_cf_scores.append((k, collab_song_infos[input_song_id][k]))

        print "INFO: Step 3, Compute closest songs:", time.time() - st,\
              ";\n \tCF data gather (", cf_data_query_time, "), Dist comp (", dist_comp_time, ")"
        return total_nn_dict, out_nn_cf_scores, total_id_list

    def get_near_neighbors_from_song_ids(self, song_ids, num_levels=1, fanout=20):
        print "**************************************************"
        print "QUERY: ", song_ids 
        print "**************************************************\n"

        t = time.time()

        song_id_list = song_ids

        final_dict = {}
        final_dict[0] = {}
        final_dict[0]['class'] = 'Root'

        # Make sure the query returned some results
        if song_id_list is not None:
            if num_levels == 1:
                nn, nn_cf_scores  = self.get_nn_one_query(song_id_list, fanout)
                final_dict[0]['children'] = nn

            else:
                queue = {}
                queue[0] = []
                nn, nn_cf_scores = self.get_nn_one_query(song_id_list, fanout)
                final_dict[0]['children'] = nn

                id_list = nn.keys()

                for n in nn.keys():
                    queue[0].append(nn[n]) 

                for level in range(num_levels-1):
                    m_nn, nn_cf_scores, id_list = self.get_nn_multi_query(id_list, nn_cf_scores, fanout)

                    for elem in queue[level]:
                        elem['children'] = m_nn[elem['song_id']] 

                    queue[level+1] = []
                    for m in m_nn.keys():
                        for k in m_nn[m].keys():
                            queue[level+1].append(m_nn[m][k]) 
        else:
            print "No songs matched the query: ", song_list
            sys.exit()

        print "----------------------------------------------------------------------------"
        print "                      QUERY PROCESSING TIME: ", time.time() - t
        print "----------------------------------------------------------------------------"

        return final_dict

    def get_near_neighbors(self, song_list, num_levels=1, fanout=20):
        print "**************************************************"
        print "QUERY: ", song_list 
        print "**************************************************\n"

        song_id_list = pardora_db_sqlite.get_song_ids_from_title_artist_pairs(song_list, self.cursor)
        print song_id_list

        final_dict = {}
        final_dict[0] = {}
        final_dict[0]['class'] = 'Root'

        # Make sure the query returned some results
        if song_id_list is not None:
            if num_levels == 1:
                nn, nn_cf_scores  = self.get_nn_one_query(song_id_list, fanout)
                final_dict[0]['children'] = nn

            else:
                queue = {}
                queue[0] = []
                nn, nn_cf_scores = self.get_nn_one_query(song_id_list, fanout)
                final_dict[0]['children'] = nn

                id_list = nn.keys()

                for n in nn.keys():
                    queue[0].append(nn[n]) 

                for level in range(num_levels-1):
                    m_nn, nn_cf_scores, id_list = self.get_nn_multi_query(id_list, nn_cf_scores, fanout)

                    for elem in queue[level]:
                        elem['children'] = m_nn[elem['song_id']] 

                    queue[level+1] = []
                    for m in m_nn.keys():
                        for k in m_nn[m].keys():
                            queue[level+1].append(m_nn[m][k]) 
        else:
            print "No songs matched the query: ", song_list
            sys.exit()

        return final_dict
    
    def print_tree(self, final_dict, levels):
        print ":::::::::::::::::::::::::::::::::::::::::::::::::::"
        print "                FINAL TREE                         "
        print "ROOT: ", final_dict[0]['class'] 

        if levels == 1:
            print "Neighbors:" 
            for nn in final_dict[0]['children'].keys():
                print "\t" + final_dict[0]['children'][nn]['artist_name'] +\
                        " - " + final_dict[0]['children'][nn]['title'] 
        elif levels == 2:
            print "Neighbors:" 
            for nn in final_dict[0]['children'].keys():
                print "\t" + final_dict[0]['children'][nn]['artist_name'] + \
                        " - " + final_dict[0]['children'][nn]['title'] 
                print "\tNeighbors:" 
                childs = final_dict[0]['children'][nn]['children']
                for c in childs.keys():
                    print "\t\t" + childs[c]['artist_name']+ " - " + childs[c]['title']
        elif levels == 3:
            print "Neighbors:" 
            for nn in final_dict[0]['children'].keys():
                print "\t" + final_dict[0]['children'][nn]['artist_name'] + " - " + final_dict[0]['children'][nn]['title'] 
                print "\tNeighbors:" 
                childs = final_dict[0]['children'][nn]['children']
                for c in childs.keys():
                    print "\t\t" + childs[c]['artist_name'] + " - " + childs[c]['title']
                    print "\t\tNeighbors:" 
                    childs2 = childs[c]['children']
                    for c2 in childs2.keys():
                        print "\t\t\t" + childs2[c2]['artist_name'] + " - " + childs2[c2]['title']
        elif levels == 4:
            print "Neighbors:" 
            for nn in final_dict[0]['children'].keys():
                print "\t" + final_dict[0]['children'][nn]['artist_name'] + " - " + final_dict[0]['children'][nn]['title'] 
                print "\tNeighbors:" 
                childs = final_dict[0]['children'][nn]['children']
                for c in childs.keys():
                    print "\t\t" + childs[c]['artist_name'] + " - " + childs[c]['title']
                    print "\t\tNeighbors:" 
                    childs2 = childs[c]['children']
                    for c2 in childs2.keys():
                        print "\t\t\t" + childs2[c2]['artist_name'] + " - " + childs2[c2]['title']
                        print "\t\t\tNeighbors:" 
                        childs3 = childs2[c2]['children']
                        for c3 in childs3.keys():
                            print "\t\t\t\t" + childs3[c3]['artist_name'] + " - " + childs3[c3]['title']

        else:
            print "Can't print tree, too many levels."

        print ":::::::::::::::::::::::::::::::::::::::::::::::::::"

    def __init__(self):
        self.conn = lite.connect(database)
        self.cursor = self.conn
        self.timbre_ubm_params, self.rhythm_ubm_params = pardora_ubm.get_UBM_parameters(M, from_pickle=True)
        print "------------- DONE INITIALIZING ----------"

    def __del__(self):
        self.conn.close()

p = Pardora()

t = time.time()

song_list = []
#song_list.append(("radiohead", "karma police"))
#song_list.append(("elton john", "angeline"))
#song_list.append(("lady gaga", "alejandro"))
#song_list.append(("cat stevens", "peace train"))
#song_list.append(("elton john", "candle in the wind"))
#song_list.append(("elton john", "memory of love"))
#song_list.append(("cat stevens", "moonshadow"))
#song_list.append(("jack johnson", "bubble toes"))
#song_list.append(("bill withers", "make love to your mind"))
#song_list.append(("perl jam", "black"))
song_list.append(("Foo Fighters", "Overdrive"))

final_dict = p.get_near_neighbors(song_list, 1, 5)
p.print_tree(final_dict, 1)

#with open(l3_output_pkl, 'wb') as fp:
#    pickle.dump(final_dict, fp)
#    fp.close()
#with open(l3_output_json, 'w') as fp:
#    json.dump(final_dict, fp)
#    fp.close()



print "----------------------------------------------------------------------------"
print "                           TOTAL TIME: ", time.time() - t
print "----------------------------------------------------------------------------"

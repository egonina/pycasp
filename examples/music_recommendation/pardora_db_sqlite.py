import MySQLdb as mdb
import numpy as np
from whoosh.query import *
import time
import binascii
import array
import sqlite3

SV_SIZE = 768

#=====================================
#          DB MANIPULATION 
#=====================================
# ===== SUPERVECTORS =====
def drop_sv_table(conn, db_cursor):
    q = 'DROP TABLE songs1m_sv'
    db_cursor.execute(q)
    conn.commit()

def create_sv_table(conn, db_cursor):
    """
    Creates the file and an empty table.
    """
    # creates file
    # add stuff
    q = 'CREATE TABLE IF NOT EXISTS '
    q += 'songs1m_sv (song_id CHAR(18), '
    q += 'timbre_sv MEDIUMBLOB, '
    q += 'timbre_sv_shape_0 INT, '
    q += 'rhythm_sv MEDIUMBLOB, '
    q += 'rhythm_sv_shape_0 INT, '
    q += 'p_mean_t REAL, '
    q += 'p_sigma_t REAL, '
    q += 'p_mean_r REAL, '
    q += 'p_sigma_r REAL, '
    q += 'PRIMARY KEY (song_id)) ENGINE=NDBCLUSTER DEFAULT CHARSET=utf8;'
    db_cursor.execute(q)
    # commit and close
    conn.commit()

def add_sv_and_p_vals_to_db(song_id, t_sv, r_sv, \
                            p_mean_t, p_sigma_t, \
                            p_mean_r, p_sigma_r, \
                            db_cursor):
    # build query
    q = "INSERT INTO songs1m_sv VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"

    t_sv_bin = sqlite3.Binary(t_sv)
    t_sv0 = int(t_sv.shape[0])
    
    r_sv_bin = sqlite3.Binary(r_sv)
    r_sv0 = int(r_sv.shape[0])

    insert_values = (song_id, t_sv_bin, t_sv0, r_sv_bin, r_sv0, \
                     p_mean_t, p_sigma_t, p_mean_r, p_sigma_r)
    
    db_cursor.execute(q, insert_values)

# ===== RHYTHM FEATURES =====
def drop_rhythm_table(conn, db_cursor):
    q = 'DROP TABLE songs1m_rhythm'
    db_cursor.execute(q)
    conn.commit()

def create_rhythm_table(conn, db_cursor):
    """
    Creates the file and an empty table.
    """
    # creates file
    q = 'CREATE TABLE IF NOT EXISTS '
    q += 'songs1m_rhythm (song_id CHAR(18), '
    q += 'rhythm_feats MEDIUMBLOB, '
    q += 'rhythm_shape_0 INT, '
    q += 'rhythm_shape_1 INT, '
    q += 'PRIMARY KEY (song_id)) ENGINE=NDBCLUSTER DEFAULT CHARSET=utf8;'
    db_cursor.execute(q)
    # commit and close
    conn.commit()

def add_rhythm_feats_to_db(song_id, r_feats, db_cursor):
    # build query
    q = "INSERT INTO songs1m_rhythm VALUES (%s, %s, %s, %s);"

    r_f_bin = sqlite3.Binary(r_feats)
    r_f0 = int(r_feats.shape[0])
    r_f1 = int(r_feats.shape[1])

    insert_values = (song_id, r_f_bin, r_f0, r_f1)
    
    db_cursor.execute(q, insert_values)

#=====================================
#          DB QUERYING 
#=====================================
def get_all_song_data(db_cursor):

    sql_query = "SELECT song_id, artist_name, title FROM songs" 
    res = db_cursor.execute(sql_query)
    songs= res.fetchall()

    total_list = []
    for s in songs:
        song_id = s[0]
        artist = s[1]
        title = s[2]

        total_list.append((song_id, artist, title))

    return total_list



def get_song_features_from_query(song_id_list, db_cursor):
    song_ids_str = str(song_id_list).strip('[]').replace("u", "").\
                 replace(",)", "").replace("(", "").replace(")", "")

    st = time.time()
    sql_query = "SELECT timbre_shape_0, timbre_shape_1, timbre_feats, artist_name, \
                title FROM songs WHERE song_id IN (" + song_ids_str + ")"
    res = db_cursor.execute(sql_query)
    timbre_result = res.fetchall()

    sql_query = "SELECT rhythm_shape_0, rhythm_shape_1, rhythm_feats \
                FROM songs WHERE song_id IN (" + song_ids_str + ")"
    res = db_cursor.execute(sql_query)
    rhythm_result = res.fetchall()
    print "TIME: get query song features from DB:\t", time.time() - st

    return timbre_result, rhythm_result

def get_song_sv_data(song_id, db_cursor):
    sql_query = "SELECT timbre_sv, rhythm_sv, \
                 p_mean_t, p_mean_r, p_sigma_t, p_sigma_r, song_id \
                 FROM songs WHERE song_id = '" + song_id  + "'"
    res = db_cursor.execute(sql_query)
    song_data = res.fetchall()

    s = song_data[0]
    total_dict = {}
    total_dict['q_t_sv'] = np.ndarray((SV_SIZE,),  buffer=s[0], dtype=np.float32)
    total_dict['q_r_sv'] = np.ndarray((SV_SIZE,),  buffer=s[1], dtype=np.float32)
    total_dict['p_mean_t'] = s[2]
    total_dict['p_mean_r'] = s[3]
    total_dict['p_sigma_t'] = s[4]
    total_dict['p_sigma_r'] = s[5]

    return total_dict

def get_song_svs_multi_query(song_id_list, db_cursor):
    ids = str(song_id_list).replace("u", "").replace("[", "").replace("]", "")

    sql_query = "SELECT timbre_sv, rhythm_sv, \
                 p_mean_t, p_mean_r, p_sigma_t, p_sigma_r, song_id \
                 FROM songs WHERE song_id IN (" + ids + ")"
    res = db_cursor.execute(sql_query)
    song_data = res.fetchall()

    total_dict = {}
    for s in song_data:
        song_id = s[6]
        total_dict[song_id] = {}
        total_dict[song_id]['q_t_sv'] = np.ndarray((SV_SIZE,),  buffer=s[0], dtype=np.float32)
        total_dict[song_id]['q_r_sv'] = np.ndarray((SV_SIZE,),  buffer=s[1], dtype=np.float32)
        total_dict[song_id]['p_mean_t'] = s[2]
        total_dict[song_id]['p_mean_r'] = s[3]
        total_dict[song_id]['p_sigma_t'] = s[4]
        total_dict[song_id]['p_sigma_r'] = s[5]

    return total_dict

def get_cf_songs_data(collab_song_info, db_cursor):
    ids = str(collab_song_info.keys()).replace("u", "").replace("[", "").replace("]", "")

    # get all song_ids
    sql_query = "SELECT title, artist_name, song_id \
                 FROM songs WHERE song_id IN (" + ids + ")"
    res = db_cursor.execute(sql_query)
    song_titles = res.fetchall()

    sql_query = "SELECT timbre_sv, rhythm_sv, \
                 p_mean_t, p_mean_r, p_sigma_t, p_sigma_r, song_id \
                 FROM songs WHERE song_id IN (" + ids + ")"
    res = db_cursor.execute(sql_query)
    song_data = res.fetchall()

    #sql_query = "SELECT mode, tempo, artist_hottness, song_id \
    #             FROM songs WHERE song_id IN (" + ids + ")"
    #res = db_cursor.execute(sql_query)
    #song_mta = res.fetchall()

    total_dict = {}
    for s in song_data:
        song_id = s[6]
        total_dict[song_id] = {}
        total_dict[song_id]['t_sv'] = np.ndarray((SV_SIZE,),  buffer=s[0], dtype=np.float32)
        total_dict[song_id]['r_sv'] = np.ndarray((SV_SIZE,),  buffer=s[1], dtype=np.float32)
        total_dict[song_id]['p_mean_t'] = s[2]
        total_dict[song_id]['p_mean_r'] = s[3]
        total_dict[song_id]['p_sigma_t'] = s[4]
        total_dict[song_id]['p_sigma_r'] = s[5]
        total_dict[song_id]['cf_score'] = collab_song_info[song_id]

    for s in song_titles:
        song_id = s[2]
        if song_id in total_dict.keys():
            total_dict[song_id]['title'] = s[0]
            total_dict[song_id]['artist_name'] = s[1]
            total_dict[song_id]['mode'] = -1 
            total_dict[song_id]['tempo'] = -1 
            total_dict[song_id]['artist_hottness'] = -1 

    #for s in song_mta:
    #    song_id = s[3]
    #    if song_id in total_dict.keys():
    #        total_dict[song_id]['mode'] = -1 
    #        total_dict[song_id]['tempo'] = -1 
    #        total_dict[song_id]['artist_hottness'] = -1 

    return total_dict

def get_song_ids_from_title_artist_pairs(song_list, db_cursor):
    song_id_list = []
    # construct artist title query strings
    title_artist_string = '('
    for pair in song_list[:-1]:
        title_artist_string += ' title = "' + str(pair[1]) + '" AND artist_name = "' + str(pair[0]) + '") OR ('

    pair = song_list[-1]
    title_artist_string += ' title = "' + str(pair[1]) + '" AND artist_name = "' + str(pair[0]) + '")'

    sql_query = 'SELECT song_id \
                 FROM songs WHERE ' + title_artist_string 

    res = db_cursor.execute(sql_query)
    song_ids = res.fetchall()

    for s in song_ids:
        song_id_list.append(s[0])

    if len(song_id_list) > 0:
        return song_id_list
    else:
        return None

def get_timbre_features_for_song_ids(song_id_list, db_cursor):
    song_ids = str(song_id_list).strip('[]').replace("u", "").replace(",)", "").replace("(", "")
    sql_query = "SELECT timbre_feats, segments_start, timbre_shape_0, \
         timbre_shape_1, sstart_shape_0, song_id FROM songs WHERE song_id IN (" \
         + song_ids + ")"
    res = db_cursor.execute(sql_query)
    songs = res.fetchall()

def get_rhythm_features_for_song_ids(song_id_list, db_cursor):
     song_ids = str(song_id_list).strip('[]').replace("u", "").replace(",)", "").replace("(", "")
     
     st = time.time()
     sql_query = "SELECT rhythm_feats, rhythm_shape_0, rhythm_shape_1, song_id \
                  FROM songs \
                  WHERE song_id IN ("+song_ids+")"
     res = db_cursor.execute(sql_query)
     songs = res.fetchall()
     return songs

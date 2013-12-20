import BaseHTTPServer
from time import sleep
from urlparse import urlparse, parse_qs
import urllib
import urllib2
import json
from pardora_main import *

SERVER_ADDR = '128.32.35.220' 
SERVER_PORT = 32456

class SongRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
        
    def recommend_songs(self, search_string):
 
        songs = search_string[0]
        song_ids = songs.split(",")
        levels = int(search_string[1])
        fanout = int(search_string[2])
        result = p.get_near_neighbors_from_song_ids(song_ids, levels, fanout)
        p.print_tree(result, 1)

        return result
        

    def do_GET(s):
        # q is the search string, e.g. "Opeth" or "tag: pretentious-metal"
        q = s.get_song_query()
        #cb = s.get_jsonp_callback()

        # use q to come up with a list of songs to queue
        if q is not None:
            songs_to_queue = s.recommend_songs(q)
            callback = q[3] # callback is fourth parameter in tuple returned by get_song_query
            s.send_response(200)
            s.send_header("Content-type", "text/javascript")
            s.end_headers()
            #s.wfile.write(json.dumps(songs_to_queue))
            s.wfile.write(callback + "(" + json.dumps(songs_to_queue) + ")")


    def get_song_query(self):
        """Extracts and returns the song query from the request.

        Looks for a query parameter "q".
        """
        query = parse_qs(urlparse(self.path).query)

        if query:
            if 'songs' in query:
                song_list = query['songs'][0]

            if 'levels' in query:
                levels = query['levels'][0]
            else:
                levels = 1 # default 1 level of expansion

            if 'fanout' in query:
                fanout = query['fanout'][0]
            else:
                fanout = 5 # default 5 neighbors 

            if 'callback' in query:
                callback = query['callback'][0]

            return (song_list, levels, fanout, callback)
        else:
            return None

    def get_jsonp_callback(self):
        """Extracts and returns the jsonp callback function name from the request.

        Looks for a query parameter "callback".
        """
        query = parse_qs(urlparse(self.path).query)
        if 'callback' in query:
            return query['callback'][0]


song_server = BaseHTTPServer.HTTPServer((SERVER_ADDR, SERVER_PORT), SongRequestHandler)
try:
    # =============== DATA PREP PHASE =================
    p = Pardora()
    print "==== Ready to recommend songs for you! ===="
    #song_server.recommend_songs("metal")
    # launch the HTTP server
    song_server.serve_forever()
except KeyboardInterrupt:
    # quietly terminate on Ctrl-C
    pass
song_server.server_close()


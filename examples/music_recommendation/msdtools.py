import cPickle as pkl
import time
#import pyItunes #this is eric's modified pyItunes
import urllib
import os
import pdb
import re
import urllib
import time

import cPickle as pkl
import numpy as np

np.seterr(all='warn')
np.seterr(over='raise')
# scipy imports
from scipy import fftpack
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix

def load_msd(filename):
    '''
    load pickle file with MSD data
    '''
    print 'loading MSD dictionary'
    t0 = time.time()
    f = open(filename,'rb')
    msd = pkl.load(f)
    f.close()
    t = time.time() - t0
    print 'loaded MSD file in %.2f seconds\n' % t

    return msd

def gen_song_lookup(msd):
    print 'Creating MSD Lookup: dict[artist][title] = song_id\n'
    r = re.compile('[^A-Za-z0-9\s]+')
    song_lookup = {}
    for (song_id,val) in msd.iteritems():
        #artist = val['artist_name']
        #title = val['title'] 
        artist = r.sub('',val['artist_name']).lower()
        title = r.sub('',val['title']).lower()
        #print '%s - %s : %s' % (artist, title, song_id)
        if song_lookup.has_key(artist):
            song_lookup[artist][title] = song_id
        else:
            song_lookup[artist] = {}
            song_lookup[artist][title] = song_id
    return song_lookup


try: # check for pyItunes
    import pyItunes #this is eric's modified pyItunes
    def load_itunes_lib(filename):
        '''
        read xml itunes library into a dictionary 
        '''
        print 'Parsing iTunes XML Library\n'
        pl = pyItunes.XMLLibraryParser(filename)
        print 'Creating pyItunes Dictionary\n'
        Library = pyItunes.Library(pl.dictionary)

        itunes_lib = {}
        

        # this loop prints out '%artist - %title' of all songs in database,
        # and checks if every file exists
        print 'Creating iTunes Lookup dict[artist][song] = file_location\n'
        numErrors = 0
        numSongs = 0
        r = re.compile('[^A-Za-z0-9\s]+')
        for song in Library.songs:

            #print '%s - %s' % (song.artist,song.name)

            loc = urllib.url2pathname(song.location)[16:]
            if not os.path.exists(loc):
                print '\nERROR, file: %s does not exist\n' % loc
                numErrors += 1
                #raw_input('press enter to continue:')
            elif song.artist is not None and song.name is not None:
                artist = r.sub('',song.artist).lower()
                title = r.sub('',song.name).lower()

                if itunes_lib.has_key(artist):
                    itunes_lib[artist][title] = loc
                else:
                    itunes_lib[artist] = {}
                    itunes_lib[artist][title] = loc

                numSongs += 1
                #print "%s - %s" % (song.artist,song.name)


        print '\n%u total iTunes file errors' % numErrors
        print '%u total iTunes songs' % len(Library.songs)

        return itunes_lib, numSongs

    def match_songs(msd_lookup, itunes_lookup):
        matching_locations = {}
        matching_songs = {}

        matched_artists = 0
        matched_songs = 0
        for artist in itunes_lookup.iterkeys():
            if msd_lookup.has_key(artist):
                print
                print artist
                matched_artists += 1
                msd_artist = msd_lookup[artist]
                itunes_artist = itunes_lookup[artist]
                for title in itunes_lookup[artist].iterkeys():
                    if msd_artist.has_key(title):
                        print '\t%s - %s' % (artist, title)
                        matched_songs += 1
                        matching_locations[msd_artist[title]] = itunes_artist[title] #key=song_id, val=file_location
                        matching_songs[msd_artist[title]] = '%s - %s' % (artist,title)

                


        print '\nmatched %u artists' % matched_artists
        print 'out of %u MSD artists' % len(msd_lookup)
        print 'out of %u iTunes artists' % len(itunes_lookup)
        print '\nmatched %u songs' % matched_songs

        return matching_locations, matching_songs
except: 
    print 'Cannot find pyItunes.  Some functions cannot be used.'


def smoothSignal(x, window_len, window='gaussian'):
    """smooth data across rows if 2D, else just smooth 1D data
    """


    if x.shape[0] < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 2:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','gaussian']:
        raise ValueError, "Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','gaussian'"

    if window_len % 2 == 1: # if odd length
        pad = (window_len-1)/2
    else:
        pad = window_len/2
    if x.ndim == 1:
        s=np.r_[2*x[0]-x[pad:0:-1], x, 2*x[-1]-x[-2:-(pad+2):-1]]
    elif x.ndim == 2:
        s=np.vstack((2*x[0]-x[pad:0:-1], x, 2*x[-1]-x[-2:-(pad+2):-1]))


    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    elif window == 'hanning':
        w = np.hanning(window_len+2)[1:-1]
    elif window == 'gaussian':
        std = window_len/6.
        w = signal.gaussian(window_len,std)
    else:
        w = getattr(np, window)(window_len)
    #y = np.convolve(w/w.sum(), s, mode='same')
    y = signal.lfilter(w/w.sum(),1,s,axis=0)
    # remove initial padding (pad) and approximate filter delay (pad)
    y = y[2*pad:]


    return y

def distanceMatrix(features):
    '''
    computes a squared-euclidean distance matrix between each pair of feature vectors

    input:
    features - [D x N] matrix containing D-dimensional feature column vectors
    output:
    dmatrix - [N x N] distance matrix (zeroes on diagonal)
    '''

    N = features.shape[1]
    x = features
    energies = np.tile(np.sum(x * x,axis=0),[N,1])
    dots = np.dot(x.T,x)

    dmatrix = energies + energies.T - 2*dots
    dmatrix[dmatrix < 0] = 0
    dmatrix = np.sqrt(dmatrix)

    return dmatrix

def distanceArray(ref,others):
    '''
    computes a squared-euclidean distance array w.r.t 'ref'

    input:
    ref - query example, [D] array containing D-dimensional feature vector
    other - [D x N] matrix of N examples to compare query to
    output:
    darray - [N] distance array
    '''

    N = others.shape[1]

    energies = np.sum(others*others,axis=0)
    dots = np.dot(ref,others)

    darray = np.dot(ref,ref) + energies - 2*dots
    darray[darray < 0] = 0
    darray = np.sqrt(darray)

    return darray

def distanceChunk(refs,others):
    '''
    computes a squared-euclidean distance array w.r.t 'ref'

    input:
    refs - query examples, [D x R] matrix containing D-dimensional feature vectors
    other - [D x N] matrix of N examples to compare query to
    output:
    darray - [R x N] distance array
    '''

    if refs.shape[0] != others.shape[0]:
        print 'distanceChunk: vector dim error'
        raise

    N = others.shape[1]

    energies = np.sum(others*others,axis=0)
    dots = np.dot(refs.T,others)

    darray = np.sum(refs*refs,axis=0) + (energies - 2*dots).T
    darray[darray < 0] = 0
    darray = np.sqrt(darray).T

    return darray

def nextPow2(x):
    y = 1
    while (y < x):
        y *= 2

    return y

class RingOfFrames:
    '''stores frames of data in a ring buffer
    and outputs them in a matrix of frames'''
    def __init__(self,numFrames,frameSize):
        self.numFrames = int(numFrames)
        self.frameSize = int(frameSize)
        self.curr = 0
        self.count = 0
        self.data = np.zeros((2*self.numFrames,self.frameSize),order='F',dtype='float32')
    def clear(self):
        self.data[:] = 0
        self.curr = 0
        self.count = 0
    def get(self,ind):
        '''get a frame with respect to last (most recent) frame
        (0 is most recent frame)'''
        if -ind >= self.numFrames:
            raise IndexError
        ind = self.curr - 1 + ind + self.numFrames
        return self.data[ind]
    def getlast(self,num):
        if num > self.numFrames:
            raise IndexError
        ind = np.arange(self.curr-num,self.curr)
        start = self.curr-num+self.numFrames
        end = self.curr+self.numFrames
        return self.data[start:end]
    def getall(self):
        start = self.curr
        end = self.curr + self.numFrames
        return self.data[start:end]
    def getFrameRange(self,start,end):
        ''' output a range of frames w.r.t. the absolute frame number
        (not including end frame)'''
        lowLimit = self.count - self.numFrames - 1
        highLimit = self.count
        if (start < lowLimit) or (start > highLimit) or (end < lowLimit) or (end > highLimit):
            pdb.set_trace()
            return None
        if start >= end:
            return None

        # find contiguous region within buffer
        offset = self.count - self.curr
        start -= offset
        end -= offset
        while start < 0:
            start += self.numFrames
            end += self.numFrames

        return self.data[start:end]

    def set(self,frame):
        try:
            self.data[self.curr] = frame
        except ValueError:
            pdb.set_trace()

        self.data[self.curr+self.numFrames] = frame
        self.curr += 1
        self.count += 1
        if self.curr == self.numFrames:
            self.curr = 0
    def setall(self,value):
        self.data[:] = value
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return repr(self.data)

def filterWeightsTriangular(sampleRate,fftLength=512,bands=(20,800,25)):
    """
    Compute non-uniform triangular filter matrix
    input:
    bands=(lowbpm,highbpm,numbands)

    output:
    weights - matrix [bands[2],fftLength]
    """

    fs = sampleRate
    nfft = fftLength
    lofreq = bands[0] 
    hifreq = bands[1]
    nbands = bands[2]


    fft_freqs = np.linspace(0,60*fs/2.,nfft/2+1) #in bpm
    f_first = max(np.nonzero(fft_freqs >= lofreq)[0][0] - 1, 0)
    f_last = min(np.nonzero(fft_freqs <= hifreq)[0][-1] + 1, fft_freqs.size-1)
    f_scaled = np.log(fft_freqs[f_first:f_last+1])
    fidx = np.zeros(nbands+2,dtype='int') #band locations in fft_freqs
    bandedges = np.linspace(f_scaled[0],f_scaled[-1],nbands+2) #band edges in f_scaled

    # find boundaries of filters
    # bandwidth of filters must be increasing
    bandwidth = 1
    fidx[0] = f_first
    for i in range(1,fidx.size):
        fidx[i] = np.argmin(np.abs(f_scaled - bandedges[i])) + f_first
        if i > 0:
            d = fidx[i]-fidx[i-1]
            if d < bandwidth:
                fidx[i] = min(fidx[i-1] + bandwidth, fft_freqs.size-1)
            else:
                bandwidth = d

    freqs = fft_freqs[fidx]
    #print 'band edges'
    #print fidx
    #print freqs
    lower = freqs[0:-2]
    center = freqs[1:-1]
    upper = freqs[2:]

    #pdb.set_trace()
    



    nbands = np.sum(upper <= np.max(fft_freqs))
    weights = np.zeros((nbands,nfft/2+1),dtype='float32')


    for i in range(nbands):
        weights[i] = \
                ((fft_freqs > lower[i]) & (fft_freqs <= center[i])) * \
                (fft_freqs-lower[i])/(center[i]-lower[i]) + \
                ((fft_freqs > center[i]) & (fft_freqs <= upper[i])) * \
                (upper[i]-fft_freqs)/(upper[i]-center[i])

    # normalize weight by bandwidth
    row_scale = np.tile(1./np.sum(weights,axis=1),[weights.shape[1],1]).T
    weights *= row_scale


    return weights, center

def filterWeights(sampleRate,sigLength,fftLength,bands=(20,800,25),shape='gaussian'):
    """
    Compute non-uniform triangular filter matrix
    input:
    bands=(lowbpm,highbpm,numbands)

    output:
    weights - matrix [bands[2],fftLength]
    """

    fs = sampleRate
    nfft = fftLength
    lofreq = bands[0] 
    hifreq = bands[1]
    nbands = bands[2]


    fft_freqs = np.linspace(0,60*fs/2.,nfft/2+1).astype('float32') #in bpm
    f_first = max(np.nonzero(fft_freqs >= lofreq)[0][0] - 1, 0)
    f_last = min(np.nonzero(fft_freqs <= hifreq)[0][-1] + 1, fft_freqs.size-1)
    f_scaled = np.log(fft_freqs[f_first:f_last+1])
    bandedges = np.linspace(f_scaled[0],f_scaled[-1],nbands+2).astype('float32') #band edges in f_scaled
    log_bandwidth = bandedges[2] - bandedges[0]


    weights = np.zeros((nbands,nfft/2+1),dtype='float32')


    if shape is 'cosine':
        # raised cosine window
        for i in range(nbands):
            mask = np.logical_and(f_scaled >= bandedges[i], f_scaled <= bandedges[i+2])
            weights[i][mask] =  0.5 * (1 + np.cos(2*np.pi*(f_scaled[mask] - bandedges[i+1])/log_bandwidth))
    elif shape is 'gaussian':
        # std equal to 1/4 log_bandwidth
        sigma2 = (log_bandwidth**2)/16
        for i in range(nbands):
            weights[i][f_first:f_last+1] =   np.exp(-(f_scaled - bandedges[i+1])**2/(2*sigma2))

    #normalize weights by bandwidth
    row_scale = np.tile(1./np.sqrt(np.sum(weights*weights,axis=1)),[weights.shape[1],1]).T
    weights *= row_scale

    centers = np.exp(bandedges[1:-1])

    return weights, centers


class EchoNestOnsetPatterns:
    '''
    simple usage:
    op = EchoNestOnsetPatterns(params_filename)
    rhythm_coefs = op.compute_coefs(timbre_features,timbre_time,song_duration)
    # rhythm_coefs contains a *row* vector of rhythm features for each block in the song
    '''

    def __init__(self,params_file=None):
        '''
        input:
        params_file - file name of params file to load from
        '''

        if params_file is not None:
            self.load_params(params_file)
        else:
            print 'Empty EchoNextOnsetPatterns initialized.'

    def compute_coefs(self,timbre_features,timbre_time,song_duration):
        '''
        compute a subset of the pca coefficients of the onset pattern of each frame

        input:
        timbre_features - stacked row vectors
        timbre_time - start times of timbre segments
        song_duraction - length of song in seconds

        output:
        coefs - onset coefficients in rows,
        outputs 'None' if not enough timbre frames exist to compute a single onset pattern
        '''

        if self.op_basis is None:
            print 'EchoNestOnsetPatterns: compression params not yet computed.'
            print 'Run method compute_compression_params.'
            raise ValueError

        periodicities = self._compute_op(timbre_features,timbre_time,song_duration)
        if periodicities is None:
            return None
        #all_coefs = self._dct2d(periodicities - self.op_mean)
        #coefs = all_coefs[:,self.dct_coef_inds[0],self.dct_coef_inds[1]]
        coefs = self._op_transform(periodicities)

        return coefs

    def recon_from_coefs(self,coefs):
        #all_coefs = np.zeros((coefs.shape[0],self.en_num_bands,self.num_log_bands),dtype='float32')
        #all_coefs[:,self.dct_coef_inds[0],self.dct_coef_inds[1]] = coefs

        periodicities = self._op_inv_transform(coefs) 

        return periodicities

    def generate_params(self,downsample=None):
        '''
        Generate the onset pattern parameters.
        Only needed when initializing an empty class.
        Should be able to init class using an existing params_file

        input:
        downsample - number of frames to downsample to from 20 per segment [None]
        '''

        print 'Setting Up Echo Nest Onset Pattern Extractor'
        print '--------------------------------------------'

        # load EN inv PCA matrix and mean
        fp = open('EN_timbre_pca.pkl','rb')
        en_pca = pkl.load(fp)
        fp.close()

        en_pca_mean = en_pca['timbre_pca_mean']
        en_inv_pca = en_pca['timbre_inv_pca']

        en_sample_rate = 22050 # Hz
        en_frame_size = 128 # samples
        # frame size == hop size
        en_num_bands = 23 
        en_attack_frames = 10
        en_decay_frames = 10

        if downsample is not None:
            # downsample inv pca matrix and pca mean by factor of 2
            if downsample % 2 != 0:
                raise ValueError, 'EchoNextOnsetPatterns: downsample value must be even'
            num_new_frames = int(downsample)
            num_orig_frames = en_attack_frames + en_decay_frames
            print 'downsampling from %u frames per segment to %u' % (num_orig_frames,num_new_frames)
            orig_locs = np.arange(num_orig_frames)
            new_locs = np.linspace(0,num_orig_frames, num_new_frames+1)[:-1]

            inv_pca_temp = []
            for col in xrange(en_inv_pca.shape[1]):
                inv_pca_col = en_inv_pca[:,col].reshape((-1,en_num_bands))
                interp_cols = interp1d(orig_locs,inv_pca_col,axis=0)(new_locs)
                inv_pca_temp += [interp_cols.flatten()]

            en_inv_pca = np.vstack(inv_pca_temp).T

            pca_mean_col = en_pca_mean.reshape((-1,en_num_bands))
            interp_cols = interp1d(orig_locs,pca_mean_col,axis=0)(new_locs)
            en_pca_mean = interp_cols.flatten()

            downsample_factor = float(num_orig_frames)/float(num_new_frames)

            en_frame_size *= downsample_factor
            en_attack_frames /= downsample_factor
            en_decay_frames /= downsample_factor

        en_frame_time = float(en_frame_size)/float(en_sample_rate)
        

        # inv pca parameters --------------------------
        self.en_pca_mean = en_pca_mean
        self.en_inv_pca = en_inv_pca
        self.en_frame_time = en_frame_time
        self.en_num_bands = en_num_bands
        self.en_attack_frames = en_attack_frames
        self.en_decay_frames = en_decay_frames
        # ------------------------------------------------


        # onset filtering
        unsharp_mask_len = int(np.round(0.25 / en_frame_time))
        # moving average filter
        unsharp_mask_filter = np.ones(unsharp_mask_len, dtype='float32') / unsharp_mask_len
        onset_filter = -unsharp_mask_filter
        onset_filter[-1] += 1.0

        # onset filtering parameters ---------------------
        self.onset_filter = onset_filter
        # ------------------------------------------------

        # periodicity
        block_size = int(np.round(4.0 / en_frame_time))
        block_padded_size = int(np.round(6.0 / en_frame_time))
        print 'desired padded size: %g sec' % (block_padded_size * en_frame_time)
        block_padded_size = int(2**(np.ceil(np.log2(block_padded_size))))
        print 'power-2 padded size: %g sec' % (block_padded_size * en_frame_time)
        block_hop_size = int(np.round(0.75 / en_frame_time))
        block_window = np.hanning(block_size)

        low_freq = 1./1.5 #Hz
        hi_freq = 13.3 # Hz
        num_log_bands = 25
        log_weights, fc = self._filter_weights_interp(sample_rate=1./en_frame_time,
                fft_length=block_padded_size,
                bands=(low_freq,hi_freq,num_log_bands))

        # periodicity params --------------------------------------
        self.log_weights = csr_matrix(log_weights)
        self.num_log_bands = num_log_bands
        self.periodicity_centers = fc
        self.block_size = block_size
        self.block_padded_size = block_padded_size
        self.block_hop_size = block_hop_size
        self.block_window = block_window
        self.chunk_size = 3
        # ---------------------------------------------------------

        # periodicity normalization -----------
        # will be calibrated by the _calibrate method
        self.periodicity_normalization = np.ones((en_num_bands,num_log_bands),dtype='float32')
        self._calibrate()

        print 'Extraction parameters generated.'
        #print 'Run generate_compression_params on timbre data to compute dct compression params.'
        self.op_mean = None
        self.op_corr = None
        #self.coef_energy = None
        self.total_blocks = None
        #self.dct_coef_inds = None
        self.op_basis = None

    def load_params(self,params_file):
        '''
        input:
        params_file - file name of params file
        '''

        print 'Loading EchoNestOnsetPatterns params from disk'
        print '----------------------------------------------'

        # read params from disk
        fp = open(params_file,'rb')
        params = pkl.load(fp)
        fp.close()

        self.en_pca_mean = params['en_pca_mean']
        self.en_inv_pca = params['en_inv_pca']
        self.en_frame_time = params['en_frame_time']
        self.en_num_bands = params['en_num_bands']
        self.en_attack_frames = params['en_attack_frames']
        self.en_decay_frames = params['en_decay_frames']

        self.onset_filter = params['onset_filter']

        self.num_log_bands = params['num_log_bands']
        self.periodicity_centers = params['periodicity_centers']
        self.block_size = params['block_size']
        self.block_padded_size = params['block_padded_size']
        self.block_hop_size = params['block_hop_size']
        self.block_window = params['block_window']

        self.log_weights = params['log_weights']
        self.periodicity_normalization = params['periodicity_normalization']

        self.op_mean = params['op_mean']
        #self.coef_energy = params['coef_energy']
        self.op_corr = params['op_corr']
        self.total_blocks = params['total_blocks']
        #self.dct_coef_inds = params['dct_coef_inds']
        self.op_basis = params['op_basis']

        self.chunk_size = params['chunk_size']

    def save_params(self,params_file):
        '''
        save params to file
        input:
        params_file - file name to save to
        '''

        params = {}

        params['en_pca_mean'] = self.en_pca_mean
        params['en_inv_pca'] = self.en_inv_pca
        params['en_frame_time'] = self.en_frame_time
        params['en_num_bands'] = self.en_num_bands
        params['en_attack_frames'] = self.en_attack_frames
        params['en_decay_frames'] = self.en_decay_frames

        params['onset_filter'] = self.onset_filter

        params['num_log_bands'] = self.num_log_bands
        params['periodicity_centers'] = self.periodicity_centers
        params['block_size'] = self.block_size
        params['block_padded_size'] = self.block_padded_size
        params['block_hop_size'] = self.block_hop_size
        params['block_window'] = self.block_window

        params['log_weights'] = self.log_weights
        params['periodicity_normalization'] = self.periodicity_normalization

        params['op_mean'] = self.op_mean
        params['op_corr'] = self.op_corr
        #params['coef_energy'] = self.coef_energy
        params['total_blocks'] = self.total_blocks
        #params['dct_coef_inds'] = self.dct_coef_inds
        params['op_basis'] = self.op_basis


        params['chunk_size'] = self.chunk_size

        fp = open(params_file,'wb')
        pkl.dump(params,fp,protocol=-1)
        fp.close()

    def compute_compression_params(self,msd_dict):
        '''
        compute dct compression parameters
        using the timbre features from msd_dict
        '''

        print 'Computing Compression Params'

        #coef_energy = np.zeros((self.en_num_bands,self.num_log_bands),dtype='float32')
        D = self.en_num_bands * self.num_log_bands
        op_sum = np.zeros(D,dtype='float32')
        op_cross_energy = np.zeros((D,D),dtype='float32')
        num_blocks = 0

        i = 0
        for v in msd_dict.itervalues():
            print '%u of %u' % (i+1,len(msd_dict))
            artist = v['artist_name']
            title = v['title']
            timbre_features = v['segments_timbre'].T
            timbre_time = v['segments_start']
            song_duration = v['duration']

            periodicity = self._compute_op(timbre_features,timbre_time,song_duration)
            if periodicity is None:
                continue

            periodicity_unrolled = periodicity.reshape((periodicity.shape[0],-1))

            #coefs = self._dct2d(periodicity)
            #coef_energy += np.sum(coefs**2,axis=0)

            op_cross_energy += np.dot(periodicity_unrolled.T,periodicity_unrolled)
            op_sum += np.sum(periodicity_unrolled,axis=0)
            num_blocks += periodicity.shape[0]

            i += 1

        if self.total_blocks is not None:

            new_total_blocks = num_blocks + self.total_blocks
            self.op_mean = (self.op_mean*self.total_blocks + op_sum)/new_total_blocks
            #self.coef_energy = (self.coef_energy*self.total_blocks + coef_energy)/new_total_blocks
            self.op_corr = (self.op_corr*self.total_blocks + op_cross_energy)/new_total_blocks
            self.total_blocks = new_total_blocks

        else:

            total_blocks = num_blocks
            self.op_mean = op_sum/total_blocks
            #self.coef_energy = coef_energy/total_blocks
            self.op_corr = op_cross_energy/total_blocks
            self.total_blocks = total_blocks

        print 'Statistics computed for %u onset pattern blocks' % self.total_blocks

        self._compute_basis()

    def _compute_basis(self,variance_threshold=0.9):
        '''
        compute a PCA basis for onset patterns
        input:
        variance_threshold - percentage of total variance to account for
        '''

    
        op_mean = self.op_mean.reshape((-1,1))
        cross_mean = np.dot(op_mean,op_mean.T)
        
        covariance = self.op_corr - cross_mean
        covariance = (covariance + covariance.T)/2

        eigvals, eigvecs = np.linalg.eig(covariance)

        sorted_inds = np.argsort(eigvals)[::-1]

        total_var = np.sum(eigvals)
        target_var = variance_threshold * total_var

        cumulative_variance = np.cumsum(eigvals[sorted_inds])
        last_coef = np.where(cumulative_variance >= target_var)[0][0]

        basis_functions = eigvecs[:,sorted_inds[:last_coef+1]]

        print 'keeping %u coefs out of %u' % (last_coef+1,sorted_inds.size)
        print 'target var: %g, retained var: %g, total var: %g' % (target_var,cumulative_variance[last_coef],total_var)

        self.op_basis = basis_functions

        return basis_functions

    def _op_transform(self,periodicity):
        '''
        transform onset patterns to onset coefs using PCA
        '''
        D = self.en_num_bands * self.num_log_bands
        if periodicity.ndim == 3:
            N = periodicity.shape[0]
            coefs = np.dot(periodicity.reshape((N,D)) - self.op_mean, self.op_basis)
        elif periodicity.ndim == 2:
            coefs = np.dot(periodicity.reshape(D) - self.op_mean, self.op_basis) 

        return coefs


    def _op_inv_transform(self,coefs):
        '''
        inverse transform onset coefs bacl to onset patterns
        '''

        D1 = self.en_num_bands 
        D2 = self.num_log_bands

        if coefs.ndim == 2:
            N = coefs.shape[0]
            recon = (np.dot(coefs, self.op_basis.T) + self.op_mean).reshape((N,D1,D2))
        elif coefs.ndim == 1:
            recon = (np.dot(coefs, self.op_basis.T) + self.op_mean).reshape((D1,D2))

        return recon





    def _compute_op(self,timbre_features,timbre_time,song_duration):
        '''
        compute onset patterns from echo nest timbre features
        '''
        #print 'computing onset patterns with chunk size %u' % chunk_size
        #t0 = time.time()
        spectrogram = self._en_inverse_pca(timbre_features,timbre_time,song_duration)
        #t1 = time.time()
        onset_spectrogram = self._unsharp_mask(spectrogram)
        #t2 = time.time()
        periodicities = self._compute_periodicities(onset_spectrogram)
        #t3 = time.time()


        #total_time = t3 - t0
        #print '\ntotal time %g sec' % total_time
        #t_pca = t1 - t0
        #print 'inverse pca %g sec' % t_pca
        #t_unsharp = t2 - t1
        #print 'unsharp mask %g sec' % t_unsharp
        #t_periods = t3 - t2
        #print 'periodicities %g sec' % t_periods

        return periodicities

    def _en_inverse_pca(self,timbre_features,timbre_time,song_duration,fast_interp=None):
        '''
        invert Echo Nest PCA timbre transformation
        '''
        timbre_frames = np.round(timbre_time / self.en_frame_time).astype('uint32')
        duration_frames = int(np.ceil(song_duration / self.en_frame_time))

        spectral_segments = (np.dot(self.en_inv_pca,timbre_features[1:]) + timbre_features[0]).T + self.en_pca_mean
        spectral_segments = spectral_segments.reshape((-1,self.en_num_bands))

        spectrogram = np.zeros((duration_frames,self.en_num_bands),dtype='float32')
        for i in xrange(len(timbre_frames)):
            attack_start = timbre_frames[i] # inclusive
            attack_end = attack_start + self.en_attack_frames # exclusive
            decay_start = attack_end # inclusive

            if i+1 < len(timbre_frames):
                decay_end = timbre_frames[i+1] # exclusive
            else:
                decay_end = duration_frames

            segment_start = i * (self.en_attack_frames + self.en_decay_frames)
            spectrogram[attack_start:attack_end] = spectral_segments[segment_start:segment_start+self.en_attack_frames]

            # linearly interpolate decay to fill remainder of segment time
            segment_mid = segment_start + self.en_attack_frames
            decay = spectral_segments[segment_mid:segment_mid+self.en_decay_frames]
            decay = np.vstack((decay,2*decay[-1]-decay[-2])) # extrapolate linearly by one frame
            num_new_decay_frames = decay_end - decay_start
            if fast_interp == 'linear':
                slope = ((decay[-1] - decay[0])/(num_new_decay_frames+1)).reshape((1,-1))
                initial = decay[0].reshape((1,-1))
                new_decay_ind = np.arange(num_new_decay_frames+1).reshape((-1,1))
                resampled_decay = (slope * new_decay_ind) + initial

            #elif fast_interp == 'exponential':
            #    initial = decay[0].reshape((1,-1))
            #    final = decay[-1].reshape((1,-1))
            #    tau = (num_new_decay_frames+1)/(np.log(initial)-np.log(final)).reshape((1,-1))
            #    inv_tau  = 1./tau
            #    new_decay_ind = np.arange(num_new_decay_frames+1).reshape((-1,1))
            #    resampled_decay = initial * np.exp(-new_decay_ind * inv_tau)

            elif fast_interp is None:
                #decay_ind = np.arange(0,self.en_decay_frames+1).astype('float32')
                decay_ind = np.linspace(0,self.en_decay_frames,self.en_decay_frames+1)
                new_decay_ind = np.linspace(0,self.en_decay_frames,num_new_decay_frames+1).astype('float32')
                resampled_decay = interp1d(decay_ind,decay,axis=0,kind='linear')(new_decay_ind)



            spectrogram[decay_start:decay_end] = resampled_decay[:-1]

        return spectrogram

    def _unsharp_mask(self,spectrogram):
        onset_spectrogram = signal.lfilter(self.onset_filter,1,spectrogram,axis=0)
        onset_spectrogram[onset_spectrogram < 0] = 0

        return onset_spectrogram

    def _compute_periodicities(self,onset_spectrogram):
        num_frames = onset_spectrogram.shape[0]
        num_blocks = int(np.floor((num_frames-self.block_size)/self.block_hop_size))
        if num_blocks <= 0:
            return None
        if self.chunk_size <= 1:
            num_chunks = 0
        elif self.chunk_size > 1:
            num_chunks = num_blocks/self.chunk_size

        num_leftover = num_blocks - num_chunks*self.chunk_size
        start_frame = 0

        #print 'num_chunks:', num_chunks, 'num_leftover:', num_leftover

        periodicities = np.zeros((num_blocks,self.en_num_bands,self.num_log_bands),dtype='float32')

        #while start_frame + chunk_frames - 1 < num_frames:
        for c in xrange(num_chunks):
            blocks = np.zeros((self.chunk_size,self.en_num_bands,self.block_size),dtype='float32')
            for i in xrange(self.chunk_size):
                blocks[i] = onset_spectrogram[start_frame:start_frame+self.block_size].T * self.block_window
                start_frame += self.block_hop_size

            band_frames = blocks.reshape((self.chunk_size*self.en_num_bands,-1))
            fft_blocks = np.fft.rfft(band_frames,axis=1,n=self.block_padded_size).astype('complex64')
            fft_energy = (fft_blocks.real**2 + fft_blocks.imag**2)
            periodicities[c*self.chunk_size:(c+1)*self.chunk_size] = self.log_weights.dot(fft_energy.T).T.reshape((self.chunk_size,self.en_num_bands,-1))

        #while start_frame + block_size - 1 < num_frames:
        for i in xrange(num_leftover):
            block = onset_spectrogram[start_frame:start_frame+self.block_size].T * self.block_window
            fft_block = np.fft.rfft(block,axis=1,n=self.block_padded_size).astype('complex64')
            fft_energy = (fft_block.real**2 + fft_block.imag**2)
            inv_fft_block = np.fft.irfft(fft_energy,axis=1)
            periodicities[i+num_chunks*self.chunk_size] = self.log_weights.dot(fft_energy.T).T
            start_frame += self.block_hop_size

        return periodicities

    def _filter_weights(self,sample_rate,fft_length,bands=(1./1.5,13.3,25)):
        '''
        Compute log-spaced triangular filter matrix
        input:
        bands=(lofreq,highfreq,numbands)
        '''

        fs = sample_rate
        nfft = fft_length
        lofreq = bands[0] 
        hifreq = bands[1]
        nbands = bands[2]


        fft_freqs = np.linspace(0,fs/2.,nfft/2+1)
        # make sure highest freq is inclusive
        hi_ind = min(np.where(fft_freqs <= hifreq)[0][-1] + 1, len(fft_freqs)-1)
        # make sure lowest freq is inclusive
        low_ind = max(np.where(fft_freqs >= lofreq)[0][0] - 1, 0)

        if low_ind == 0:
            f = fft_freqs[low_ind+1:hi_ind+1]
            log_f = np.log(f)
            f_ind = np.zeros(nbands+1,dtype='int')
            log_targets = np.linspace(log_f[0],log_f[-1],nbands+1)
        else:
            f = fft_freqs[low_ind:hi_ind+1]
            log_f = np.log(f)
            f_ind = np.zeros(nbands+2,dtype='int')
            log_targets = np.linspace(log_f[0],log_f[-1],nbands+2)


        # find boundaries of filters
        bandwidth = 1
        for i in range(0,f_ind.size):

            if i == 0:
                f_ind[i] = 0
            else:
                f_ind[i] = f_ind[i-1]

            # increase index until target is surpassed
            while log_f[f_ind[i]] < log_targets[i]:
                f_ind[i] += 1

            # check if previous index results in less error
            if f_ind[i] > 0:
                error0 = np.abs(log_f[f_ind[i]] - log_targets[i])
                error1 = np.abs(log_f[f_ind[i]-1] - log_targets[i])
                if error0 > error1:
                    f_ind[i] -= 1

            # make sure bandwidth is monotonically increasing
            if i > 0:
                d = f_ind[i]-f_ind[i-1]
                if d < bandwidth:
                    f_ind[i] = min(f_ind[i-1] + bandwidth, f.size)
                else:
                    bandwidth = d

        if low_ind == 0:
            # add dc component to front of band edges
            freqs = np.insert(f[f_ind], 0, fft_freqs[low_ind])
            lower = freqs[0:-2]
            center = freqs[1:-1]
            upper = freqs[2:]
        else:
            freqs = f[f_ind]
            lower = freqs[0:-2]
            center = freqs[1:-1]
            upper = freqs[2:]

        #nbands = np.sum(upper <= np.max(fft_freqs))
        weights = np.zeros((nbands,nfft/2+1))


        # construct triangular-shaped filters
        for i in range(nbands):
            weights[i] = \
                    ((fft_freqs > lower[i]) & (fft_freqs <= center[i])) * \
                    (fft_freqs-lower[i])/(center[i]-lower[i]) + \
                    ((fft_freqs > center[i]) & (fft_freqs <= upper[i])) * \
                    (upper[i]-fft_freqs)/(upper[i]-center[i])

        return weights, center

    def _filter_weights_interp(self,sample_rate,fft_length,bands=(1./1.5,13.3,25)):
        '''
        Compute log-spaced triangular filter matrix
        input:
        bands=(lofreq,highfreq,numbands)
        '''

        fs = sample_rate
        nfft = fft_length
        lofreq = bands[0] 
        hifreq = bands[1]
        nbands = bands[2]


        fft_freqs = np.linspace(0,fs/2.,nfft/2+1)
        fft_freqs[0] = np.finfo('float32').tiny

        log_freqs = np.log(fft_freqs)
        log_centers = np.linspace(np.log(lofreq),np.log(hifreq),nbands)
        log_bandwidth = log_centers[1] - log_centers[0]
        # gaussian decays to 0.5 after half bandwidth
        sigma2 = log_bandwidth**2 / np.log(16)

        center = np.exp(log_centers)

        weights = np.zeros((nbands,nfft/2+1))

        # construct triangular-shaped filters
        for i in range(nbands):
            weights[i] = np.exp(-(log_freqs-log_centers[i])**2/sigma2)
            weights[i][weights[i] < 0.1*weights[i].max()] = 0

        weights /= fft_length

        return weights, center

    def _calibrate(self):
        print 'Calibrating Echo Nest onset pattern extractor'

        calibration_time = 10 # sec
        calibration_frames = float(calibration_time) / self.en_frame_time
        t = np.arange(calibration_frames)
        new_normalization = np.zeros_like(self.periodicity_normalization)
        for i in xrange(self.num_log_bands):
            freq = self.periodicity_centers[i]
            calibration_signal = 1 + np.cos(2*np.pi*freq*t*self.en_frame_time)
            calibration_spectrogram = np.tile(calibration_signal,[self.en_num_bands,1]).T

            onset_spectrogram = self._unsharp_mask(calibration_spectrogram)
            periodicities = self._compute_periodicities(onset_spectrogram)

            mean_value = np.mean(periodicities[:,0,i]) # 0 - all bands are equal
            new_normalization[:,i] = 1./mean_value

        self.periodicity_normalization = new_normalization / new_normalization.max()
        self.log_weights = csr_matrix(self.periodicity_normalization[0].reshape((-1,1)) * self.log_weights.toarray())

    def _dct2d(self,periodicities,norm='ortho'):
        '''
        compute 2d type II DCT along axis=1,2
        
        dct is separable so we compute it using two 1-d dct's

        input:
        norm - {'ortho',None}
        '''

        if periodicities.ndim == 3:
            x_dct = fftpack.dct(periodicities,axis=1,norm=norm)
            y_dct = fftpack.dct(x_dct,axis=2,norm=norm)
        elif periodicities.ndim == 2:
            x_dct = fftpack.dct(periodicities,axis=0,norm=norm)
            y_dct = fftpack.dct(x_dct,axis=1,norm=norm)


        return y_dct

    def _idct2d(self,coeffs,norm='ortho'):
        '''
        compute inverse 2d type II DCT along axis=1,2

        input:
        norm - {'ortho',None}
        '''


        if coeffs.ndim == 3:
            y_idct = fftpack.idct(coeffs,axis=2,norm=norm)
            x_idct = fftpack.idct(y_idct,axis=1,norm=norm)
        elif coeffs.ndim == 2:
            y_idct = fftpack.idct(coeffs,axis=1,norm=norm)
            x_idct = fftpack.idct(y_idct,axis=0,norm=norm)


        return x_idct


def rhythm_features(seg_timbre,seg_time):
    '''
    compute rhythm features from timbre features at onset locations from MSD.
    features are computed synchronously with constant window hopsize

    input:
    seg_timbre - [N x D] timbre feature matrix where N is number of analysis segments
    seg_time - length-N array containing onset time of timbre segment

    global dependencies (loaded from disk)
    rhythm_params - dictionary of rhythm extraction parameters

    output:
    onset_spectrum - [F x H] feature matrix where H is number of analysis frames
    onset_pattern - average onset_specrum for entire song
    '''

    # make sure rhythm_params were loaded from disk
    if rhythm_params_loaded is not True:
        print 'rhythm_params not loaded from disk'
        raise

    frame_rate = 1./frame_width
    window_hop_frames = window_width_frames/hops_per_window


    #frame_width = 1024/44100. #23ms

    #window_width = 3 #sec
    #hops_per_window = 4 
    #window_width_frames = hops_per_window * int(np.round(window_width / frame_width / hops_per_window))
    #window_hop_frames = window_width_frames/hops_per_window
    #max_lag = 3 #sec
    #max_lag_frames = int(np.round(max_lag / frame_width))

    seg_frame = np.round(seg_time * frame_rate).astype('int')

    window_start_frame = 0
    start_idx = 0

    # compute start and end segment indices for each window
    start_indices = []
    window_start_frames = []
    start_idx = 0
    while window_start_frame + window_hop_frames + max_lag_frames < seg_frame[-1]:
        while seg_frame[start_idx] < window_start_frame:
            start_idx += 1
        start_indices.append(start_idx)
        window_start_frames.append(window_start_frame)
        window_start_frame += window_hop_frames

    if len(window_start_frames) == 0:
        return None,None

    onset_corr = np.zeros((max_lag_frames,len(start_indices)),dtype='float32')
    corr_ring = RingOfFrames(hops_per_window,max_lag_frames)
    hop_scaling = np.hanning(hops_per_window+2)[1:-1].astype('float32')
    hop_scaling = np.tile(hop_scaling,[max_lag_frames,1]).T
    window_number = 0

    #normalize average squared 2-norm to be 1 
    E2 = np.sum(seg_timbre * seg_timbre)/seg_timbre.shape[0]
    alpha = 1./E2


    for i in range(len(start_indices)):

        corr_local = np.zeros((max_lag_frames,),dtype='float32')
        start_idx = start_indices[i]
        window_start_frame = window_start_frames[i]


        while seg_frame[start_idx] <  window_start_frame + window_hop_frames:
            end_idx = start_idx
            while seg_frame[end_idx] < seg_frame[start_idx] + max_lag_frames:
                if end_idx+1 >= len(seg_frame):
                    end_idx += 1
                    break
                else:
                    end_idx += 1

            lags = seg_frame[start_idx:end_idx] - seg_frame[start_idx]

            similarity = alpha * np.dot(seg_timbre[start_idx:end_idx],seg_timbre[start_idx])

            corr_local[lags] += similarity

            start_idx += 1

        corr_ring.set(corr_local)
        onset_corr[:,window_number] = np.sum(corr_ring.getall() * hop_scaling,axis=0)
        window_number += 1

    # use single-precision real fft
    spectrum = fftpack.rfft(onset_corr,n=fftLength,axis=0)
    spectrum_energy = np.zeros((fftLength/2+1,spectrum.shape[1]),dtype='float32')
    spectrum_energy[0] = spectrum[0]**2
    spectrum_energy[-1] = spectrum[-1]**2
    spectrum_energy[1:-1] = spectrum[1:-1:2]**2 + spectrum[2:-1:2]**2

    onset_spectrum = np.sqrt(np.dot(log_weights, spectrum_energy))
    onset_coefs = np.dot(rhythm_pca,(onset_spectrum.T-rhythm_mean).T) #pca dim-reduce
    onset_pattern = np.median(onset_coefs,axis=1) #song summary

    if np.sum(onset_coefs) == 0:
        pdb.set_trace()


    return onset_coefs, onset_pattern

def mcs_norm(features,meanFeatures=0):
    '''
    normalize features using mcs-norm
    input: features - [D x N] d-dimensional features, N feature vectors
    output: [D x N] but normalized
    '''

    x = features.copy().T
    #pdb.set_trace()
    x -= meanFeatures

    if x.ndim > 1:
        scale = 1./np.sqrt(np.sum(x*x,axis=1))
    else:
        scale = 1./np.sqrt(np.sum(x*x))
    return scale * x.T

def p_norm_params_single(ref, others):
    '''
    compute p-norm params (mean and std)
    using distance stats between ref and others

    input:
    ref - reference vector
    others - test comparison vectors

    output:
    mu - mean distance
    std - std of distance
    '''
    da = distanceArray(ref, others)

    mu = np.mean(da)
    sigma = np.std(da)

    return mu, sigma

def p_norm_params_chunk(refs, others, chunk_size=1):
    '''
    compute p-norm params (mean and std)
    using distance stats between ref and others
    --computation is done in chunks that fit into memory

    input:
    ref - [D x N] reference vectors
    others - [D x M] test comparison vectors
    chunk_size - processing blocksize

    output:
    mu - mean distance
    std - std of distance
    '''

    N = refs.shape[1]

    if refs.shape[0] != others.shape[0]:
        print 'error: vector dimension mismatch'
        raise

    if chunk_size > N:
        chunk_size = N

    chunks = N/chunk_size
    leftover = N - chunks*chunk_size

    mu = np.zeros((N,),dtype='float32')
    sigma = mu.copy()

    for i in range(chunks):
        print 'p-norm params: chunk %u+1 of %u' % (i,chunks) 
        rng = slice(i*chunk_size,(i+1)*chunk_size)
        da = distanceChunk(refs[:,rng], others)

        mu[rng] = np.mean(da,axis=1)
        sigma[rng] = np.std(da,axis=1)

    if leftover > 0:
        rng = slice(chunks*chunk_size,chunks*chunk_size+leftover)
        da = distanceChunk(refs[:,rng], others)

    mu[rng] = np.mean(da,axis=1)
    sigma[rng] = np.std(da,axis=1)
    
    return mu, sigma

def p_norm_params(features):
    '''
    compute p-norm params (means and std's)
    using distance stats between each pair of feature vectors

    input:
    features - [D x N] matrix of D-dim feature vectors

    output:
    mu - mean distances
    std - std of distances
    '''

    da = distanceMatrix(features)

    mu = np.mean(da,axis=1)
    sigma = np.std(da,axis=1)

    return mu, sigma

def p_norm_distance(features, mu, sigma):
    '''
    compute a p-normed distance array between pairs of features

    input:
    features - [D x N] matrix of D-dim feature vectors
    mu - array of p-norm means
    sigma - array of p-norm std's

    output:
    pdist - p-normed distance matrix between ref and others
    '''

    da = distanceMatrix(features)
    da -= mu
    da *= (1./sigma)

    pdist = 0.5 * (da + da.T)

    return pdist

def p_norm_distance_single(ref, others, ref_mean, others_mean, ref_std, others_std):
    '''
    compute a p-normed distance array between ref and others

    input:
    ref - target feature vector
    others - [D x N] matrix of D-length features vectors
    ref_mean - p-norm mean for ref
    others_mean - array of p-norm means for others
    ref_std - p-norm std for ref
    others_std - array of p-norm std's for others

    output:
    pdist - p-normed distance array between ref and others
    '''

    da = np.array(distanceArray(ref, others), dtype=np.float32)

    pdist = 0.5 * ( (da - ref_mean)*(1./ref_std) + (da - others_mean)/others_std )

    return pdist

def p_norm_distance_chunk(refs, others, refs_mean, others_mean, refs_std, others_std, chunk_size=1):
    '''
    compute a p-normed distance matrix between refs and others
    using distance stats between refs and others
    --computation is done in chunks that fit into memory

    input:
    ref - [D x N] reference vectors
    others - [D x M] test comparison vectors
    refs_mean - [N] p-norm means
    others_mean - [M] p-norm means
    refs_std - [N] p-norm std's
    others_std - [M] p-norm std's
    chunk_size - processing blocksize

    output:
    pdist - [N x M] distance matrix
    '''

    N = refs.shape[1]
    M = others.shape[1]

    if refs.shape[0] != others.shape[0]:
        print 'error: vector dimension mismatch'
        raise

    if chunk_size > N:
        chunk_size = N

    chunks = N/chunk_size
    leftover = N - chunks*chunk_size

    pdist = np.zeros((N,M),dtype='float32')

    for i in range(chunks):
        print 'p-norm distance: chunk %u+1 of %u' % (i,chunks) 
        rng = slice(i*chunk_size,(i+1)*chunk_size)
        da = distanceChunk(refs[:,rng], others)

        pdist[rng] = 0.5 * ( ((da.T - refs_mean[rng])*(1./refs_std[rng])).T + (da - others_mean)*(1./others_std) )

    if leftover > 0:
        rng = slice(chunks*chunk_size,chunks*chunk_size+leftover)
        da = distanceChunk(refs[:,rng], others)
        pdist[rng] = 0.5 * ( ((da.T - refs_mean[rng])*(1./ref_std[rng])).T + (da - others_mean)*(1./others_std) )

    
    return pdist

    







#load rhythm_features() params
try:
    print "Loading rhythm params"
    fp = open('/disk1/home_user/egonina/msd_database/pickles/rhythm_params.pkl','rb')
    rhythm_params = pkl.load(fp)
    fp.close()

    frame_width = rhythm_params['frame_width']
    window_width_frames = rhythm_params['window_width_frames']
    fftLength = rhythm_params['fftLength']
    max_lag_frames = rhythm_params['max_lag_frames']
    hops_per_window = rhythm_params['hops_per_window']
    log_weights = rhythm_params['log_weights']['weights']
    rhythm_pca = rhythm_params['onset_spectrum_stats']['eigvec'][:,:12].T
    rhythm_mean = rhythm_params['onset_spectrum_stats']['mean']
    #timbre_stats = rhythm_params['timbre_stats']

    rhythm_params_loaded = True
except:
    print 'error processing rhythm_params.pkl'
    rhythm_params_loaded = False


def rhythm_features_gen_params():
    '''
    generate a parameters file for rhythm_features() function
    '''

    # setup rhythm parameters from file loaded from disk
    #frame_width = rhythm_params['frame_width']
    #frame_rate = 1./frame_width
    #window_width_frames = rhythm_params['window_width_frames']
    #fftLength = rhythm_params['fftLength']
    #max_lag_frames = rhythm_params['max_lag_frames']
    #hops_per_window = rhythm_params['hops_per_window']
    #window_hop_frames = window_width_frames/hops_per_window
    #log_weights = rhythm_params['log_weights']
    #timbre_cov_inv = rhythm_params['timbre_cov_inv']
    #pca_matrix = rhythm_params['pca_matrix']

    frame_width = 1024/44100. #23ms

    window_width = 3 #sec
    padded_window_width = 6 #sec
    hops_per_window = 4 
    max_lag = 3 #sec

    window_width_frames = hops_per_window * int(np.round(window_width / frame_width / hops_per_window))
    window_hop_frames = window_width_frames/hops_per_window
    max_lag_frames = int(np.round(max_lag / frame_width))
    fftLength = nextPow2(int(np.round(max(padded_window_width / frame_width, max_lag_frames))))

    mahalanobis = True

    min_bpm = 20
    max_bpm = 800
    num_bands = 25
    weights, centers = filterWeights(1./frame_width,max_lag_frames,fftLength,(min_bpm,max_bpm,num_bands))

    # load matrix from disk
    fp = open('timbre_stats.pkl','rb')
    timbre_stats = pkl.load(fp)
    fp.close()
    fp = open('onset_spectrum_stats.pkl','rb')
    onset_spectrum_stats = pkl.load(fp)
    fp.close()

    rhythm_params = {'frame_width':frame_width, 'window_width_frames':window_width_frames, 
            'fftLength':fftLength,
            'max_lag_frames':max_lag_frames, 'hops_per_window':hops_per_window, 
            'log_weights':{'weights':weights,'centers':centers},
            'timbre_stats':timbre_stats,
            'onset_spectrum_stats':onset_spectrum_stats,
            }

    name = raw_input('filename to save rhythm params [rhythm_params.pkl]:')
    if name is '':
        name = 'rhythm_params.pkl'
        
    fp = open(name,'wb')
    pkl.dump(rhythm_params,fp,protocol=-1)
    fp.close()

            
    return rhythm_params


def rhythm_features_mahalanobis(seg_timbre,seg_time):
    '''
    compute rhythm features from timbre features at onset locations from MSD.
    features are computed synchronously with constant window hopsize

    input:
    seg_timbre - [N x D] timbre feature matrix where N is number of analysis segments
    seg_time - length-N array containing onset time of timbre segment

    global dependencies (loaded from disk)
    rhythm_params - dictionary of rhythm extraction parameters

    output:
    onset_spectrum - [D x N] feature matrix where N is number of analysis frames
    onset_pattern - average onset_specrum for entire song
    '''

    # make sure rhythm_params were loaded from disk
    try:
        if rhythm_params_loaded is not True:
            raise
    except:
        print 'rhythm_params not loaded from disk'
        return None, None

    frame_rate = 1./frame_width
    window_hop_frames = window_width_frames/hops_per_window


    #frame_width = 1024/44100. #23ms

    #window_width = 3 #sec
    #hops_per_window = 4 
    #window_width_frames = hops_per_window * int(np.round(window_width / frame_width / hops_per_window))
    #window_hop_frames = window_width_frames/hops_per_window
    #max_lag = 3 #sec
    #max_lag_frames = int(np.round(max_lag / frame_width))

    seg_frame = np.round(seg_time * frame_rate,dtype='float32')

    window_start_frame = 0
    start_idx = 0

    # compute start and end segment indices for each window
    start_indices = []
    window_start_frames = []
    start_idx = 0
    while window_start_frame + window_hop_frames + max_lag_frames < seg_frame[-1]:
        while seg_frame[start_idx] < window_start_frame:
            start_idx += 1
        start_indices.append(start_idx)
        window_start_frames.append(window_start_frame)
        window_start_frame += window_hop_frames

    if len(window_start_frames) == 0:
        return None,None

    onset_corr = np.zeros((max_lag_frames,len(start_indices)),dtype='float32')
    corr_ring = RingOfFrames(hops_per_window,max_lag_frames)
    hop_scaling = np.hanning(hops_per_window+2)[1:-1].astype('float32')
    hop_scaling = np.tile(hop_scaling,[max_lag_frames,1]).T
    window_number = 0

    mahalanobis = True
    if mahalanobis:
        # normalize the expected squared Mahalanobis distance between
        # timbre frames to be 'T'
        T = np.log(2)
        Ed = 2 * timbre_stats['invcov'].shape[0]
        alpha = T/Ed
    else:
        # normalize the expected squared distance between timbre frames
        # to be 'T'
        T = np.log(2)
        Ed = 2 * np.sum(timbre_stats['cov'].diagonal())
        alpha = T/Ed


    for i in range(len(start_indices)):

        corr_local = np.zeros((max_lag_frames,),dtype='float32')
        start_idx = start_indices[i]
        window_start_frame = window_start_frames[i]


        while seg_frame[start_idx] <  window_start_frame + window_hop_frames:
            end_idx = start_idx
            while seg_frame[end_idx] < seg_frame[start_idx] + max_lag_frames:
                if end_idx+1 >= len(seg_frame):
                    end_idx += 1
                    break
                else:
                    end_idx += 1
            lags = seg_frame[start_idx:end_idx] - seg_frame[start_idx]

            #cross_term = np.dot(seg_timbre[start_idx:end_idx],seg_timbre[start_idx])
            #norms = np.sqrt(np.sum(seg_timbre[start_idx:end_idx]**2,axis=1))
            #cosine_similarity = cross_term / norms / norms[0]

            diffs = seg_timbre[start_idx]-seg_timbre[start_idx:end_idx]
            
            if mahalanobis:
                distance =  alpha * np.sum(np.dot(diffs,timbre_cov_inv) * diffs, axis=1)
            else:
                distance =  alpha * np.sum(diffs**2, axis=1)




            #corr_local[lags] += 1
            #corr_local[lags] += cosine_similarity
            #corr_local[lags] += 1/(distance + 1)
            corr_local[lags] += np.exp(-distance)

            start_idx += 1

        corr_ring.set(corr_local)
        onset_corr[:,window_number] = np.sum(corr_ring.getall() * hop_scaling,axis=0)
        window_number += 1



    #smooth_window_len = max(int(np.round(frame_rate * 2 * 0.020)), 3)
    #onset_corr = smoothSignal(onset_corr,smooth_window_len)

    #padded_window_width = 6 #sec
    #fftLength = nextPow2(int(np.round(padded_window_width / frame_width)))

    # use single-precision real fft
    spectrum = fftpack.rfft(onset_corr,n=fftLength,axis=0)
    spectrum_energy = np.zeros((fftLength/2+1,spectrum.shape[1]),dtype='float32')
    spectrum_energy[0] = spectrum[0]**2
    spectrum_energy[-1] = spectrum[-1]**2
    spectrum_energy[1:-1] = spectrum[1:-1:2]**2 + spectrum[2:-1:2]**2

    #log_weights, centers = filterWeights(frame_rate,fftLength,(20,800,25))

    onset_spectrum = np.sqrt(np.dot(log_weights, spectrum_energy))
    onset_pattern = np.median(onset_spectrum,axis=1)

    pdb.set_trace()

    # perform pca 
    #onset_spectrum = np.dot(os_pca, onset_spectrum - np.tile(os_mean,[onset_spectrum.shape[1],1]).T)
    #onset_pattern = np.dot(op_pca, onset_pattern - op_mean)


    return onset_spectrum, onset_pattern


def rhythm_features_async_bak(seg_timbre,seg_time,timbre_cov_inv):
    '''
    compute rhythm features from timbre features at onset locations from MSD
    _async: features are computed for analysis windows starting at every onset

    input:
    seg_timbre - timbre feature for onset segment
    seg_time - onset time of segment
    timbre_cov_inv - inverse of covariance matrix for timbre features

    output:

    '''

    frame_width = 1024/44100. #23ms

    window_width = 3 #sec
    hops_per_window = 4 
    window_width_frames = hops_per_window * int(np.round(window_width / frame_width / hops_per_window))
    window_hop_frames = window_width_frames/hops_per_window
    max_lag = 3 #sec
    max_lag_frames = int(np.round(max_lag / frame_width))

    seg_frame = np.round(seg_time * (1./frame_width),dtype='float32')

    window_start_frame = 0
    start_idx = 0

    # compute start and end segment indices for each window
    start_indices = []
    start_idx = 0
    while seg_frame[start_idx] + window_width_frames - 1 + max_lag_frames - 1 <= seg_frame[-1]:
        start_indices.append(start_idx)
        start_idx += 1

    if len(start_indices) == 0:
        return None,None

    onset_corr = np.zeros((max_lag_frames,len(start_indices)),dtype='float32')
    #corr_ring = RingOfFrames(hops_per_window,max_lag_frames)
    #hop_scaling = np.hanning(hops_per_window+2)[1:-1].astype('float32')
    #hop_scaling = np.tile(hop_scaling,[max_lag_frames,1]).T
    #window_number = 0

    for i in range(len(start_indices)):

        corr_local = np.zeros((max_lag_frames,),dtype='float32')
        start_idx = start_indices[i]
        first_frame = start_idx

        while seg_frame[start_idx] <  first_frame + window_width_frames:
            end_idx = start_idx
            while seg_frame[end_idx] < seg_frame[start_idx] + max_lag_frames:
                end_idx += 1
                #if end_idx+1 >= len(seg_frame):
                #    end_idx += 1
                #    break
                #else:
                #    end_idx += 1
            lags = seg_frame[start_idx:end_idx] - seg_frame[start_idx]

            #cross_term = np.dot(seg_timbre[start_idx:end_idx],seg_timbre[start_idx])
            #norms = np.sqrt(np.sum(seg_timbre[start_idx:end_idx]**2,axis=1))
            #cosine_similarity = cross_term / norms / norms[0]

            diffs = seg_timbre[start_idx]-seg_timbre[start_idx:end_idx]
            distance =  np.mean(np.dot(diffs,timbre_cov_inv) * diffs, axis=1)


            #corr_local[lags] += 1
            #corr_local[lags] += cosine_similarity
            #corr_local[lags] += 1/(distance + 1)
            corr_local[lags] += np.exp(-distance)

            start_idx += 1

        #corr_ring.set(corr_local)
        #onset_corr[:,i] = np.sum(corr_ring.getall() * hop_scaling,axis=0)
        onset_corr[:,i] = corr_local

    frame_rate = 1./frame_width
    #smooth_window_len = int(np.round(frame_rate * 2 * 0.020))
    #onset_corr = smoothSignal(onset_corr,smooth_window_len)

    padded_window_width = 6 #sec
    fftLength = nextPow2(int(np.round(padded_window_width / frame_width)))

    # use single-precision real fft
    spectrum = fftpack.rfft(onset_corr,n=fftLength,axis=0)
    spectrum_energy = np.zeros((fftLength/2+1,spectrum.shape[1]),dtype='float32')
    spectrum_energy[0] = spectrum[0]**2
    spectrum_energy[-1] = spectrum[-1]**2
    spectrum_energy[1:-1] = spectrum[1:-1:2]**2 + spectrum[2:-1:2]**2

    log_weights, centers = filterWeights(frame_rate,fftLength,(20,800,25))

    onset_spectrum = np.sqrt(np.dot(log_weights, spectrum_energy))

    #onset_coefs = fftpack.dct(onset_spectrum,axis=0)[:12]
    onset_pattern = np.median(onset_spectrum,axis=1)


    return onset_pattern, onset_spectrum


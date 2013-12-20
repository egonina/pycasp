from mrjob.job import  MRJob
from mrjob.protocol import PickleProtocol as protocol

import sys
import time
import ConfigParser
from diarizer import *

class ClusterMRJob(MRJob):
        
    INPUT_PROTOCOL = protocol
    OUTPUT_PROTOCOL = protocol
    
    def job_runner_kwargs(self):
        config = super(ClusterMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] =  "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 1
        config['cmdenv']['PYTHONPATH'] = ":".join([
            "/n/shokuji/da/penpornk/all"
        ])
        config['cmdenv']['~'] = "/n/shokuji/da/penpornk"
        config['cmdenv']['HOME'] = "/n/shokuji/da/penpornk"
        config['cmdenv']['MPLCONFIGDIR'] = "/n/shokuji/da/penpornk"
        config['cmdenv']['PATH'] = ":".join([
            "/n/shokuji/da/penpornk/env/gmm/bin",
            "/n/shokuji/da/penpornk/local/bin",
            "/usr/local/bin", "/usr/bin", "/bin",
            "/usr/X11/bin",
            "/usr/local64/lang/cuda-3.2/bin/",
            "/n/shokuji/da/penpornk/local/hadoop/bin"
        ])
        config['cmdenv']['LD_LIBRARY_PATH'] = ":".join([
            "/usr/lib64/atlas",
            "/usr/local64/lang/cuda-3.2/lib64",
            "/usr/local64/lang/cuda-3.2/lib",
            "/n/shokuji/da/penpornk/local/lib"                                            
        ])
        config['cmdenv']['BLAS'] = "/usr/lib64/atlas/libptcblas.so"
        config['cmdenv']['LAPACK'] = "/usr/lib64/atlas/liblapack.so"
        config['cmdenv']['ATLAS'] = "/usr/lib64/atlas/libatlas.so"
        config['cmdenv']['C_INCLUDE_PATH'] = "/n/shokuji/da/penpornk/local/include"
        config['cmdenv']['CPLUS_INCLUDE_PATH'] = "/n/shokuji/da/penpornk/local/include"
        config['python_bin'] = "/n/shokuji/da/penpornk/env/gmm/bin/python"
        config['bootstrap_mrjob'] = False
        return config
        
    def hadoop_job_runner_kwargs(self):
        config = super(ClusterMRJob, self).hadoop_job_runner_kwargs()
        config['hadoop_extra_args'] += [
            "-verbose",
        #    "-mapdebug", "/n/shokuji/da/penpornk/diarizer/debug.sh"
        ]
        return config
    
    def mapper(self, key, _):
        device_id = 0
        start = time.time()

        # update with the config for the diarizer and log file name
        config_file = 'filename.cfg'.format(key)
        logfile = 'filename.log'.format(key)
        log = open(logfile, 'w')
        tmp = sys.stdout
        sys.stdout = log
    
        try:
            open(config_file)
        except IOError, err:
            print >> sys.stderr, "Error! Config file: '", config_file, "' does not exist"
            sys.exit(2)
            
        # Parse diarizer config file
        config = ConfigParser.ConfigParser()
    
        config.read(config_file)
    
        meeting_name, f, sp, outfile, gmmfile, num_gmms, num_comps, num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters, seg_length = get_config_params(config)        
        print >> sys.stderr, meeting_name

        
        # Create tester object
        diarizer = Diarizer(f, sp)
    
        # Create the GMM list
        diarizer.new_gmm_list(num_comps, num_gmms, 'diag')
    
        # Cluster
        most_likely = diarizer.cluster(num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters, seg_length)
    
        # Write out RTTM and GMM parameter files
        diarizer.write_to_RTTM(outfile, sp, meeting_name, most_likely, num_gmms, seg_length)
        diarizer.write_to_GMM(gmmfile)
        
        print >> sys.stderr, "Time:", time.time()-start
        sys.stdout = tmp
        log.close()
        yield 1, 1

if __name__ == '__main__':
    ClusterMRJob.run()

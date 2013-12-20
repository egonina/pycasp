import time
import os.path

from cluster_map import ClusterMRJob
from mrjob.protocol import PickleProtocol as protocol

 
# video file names, not all included for brevity
all_meeting_names = ['E002/HVC686083', 'E002/HVC690403', 'E002/HVC693882', 'E002/HVC698972', 'E002/HVC699015', 'E002/HVC724149', 'E002/HVC725414', 'E002/HVC732985', 'E002/HVC738245', 'E002/HVC742544', 'E002/HVC765357', 'E002/HVC765442', 'E002/HVC768066', 'E002/HVC781040', 'E002/HVC781550', 'E002/HVC787358', 'E002/HVC788059', 'E002/HVC792707', 'E002/HVC800578', 'E002/HVC804688', 'E002/HVC805254', 'E002/HVC805769', 'E002/HVC820877', 'E002/HVC826976', 'E002/HVC827162', 'E002/HVC831492', 'E002/HVC843356', 'E002/HVC843604', 'E002/HVC848042', 'E002/HVC852977', 'E002/HVC860018', 'E002/HVC860974', 'E002/HVC862982', 'E002/HVC868918', 'E002/HVC875036', 'E002/HVC876313', 'E002/HVC885657', 'E002/HVC893844']

def preprocess(names):
    tuples = map(lambda(x): (x, os.path.getsize('/u/drspeech/data/Aladdin/corpora/trecvid2011/events/'+x+'.htk')), names)
    return map(lambda(name, size): name, sorted(tuples, key=lambda(x): x[1], reverse=True))

if __name__ == '__main__':
    mr_args = ['-v', '--strict-protocols', '-r', 'hadoop','--input-protocol', 'pickle','--output-protocol','pickle','--protocol','pickle']
    meeting_names = all_meeting_names[:250]
    meeting_names = preprocess(meeting_names)
    print "Processing {0} input files".format(len(meeting_names))
    task_args = [protocol.write(name, None)+"\n" for name in meeting_names]
    
    start = time.time()
    job = ClusterMRJob(args=mr_args).sandbox(stdin=task_args)
    runner = job.make_runner()        
    runner.run()

    print "Tasks done. Total execution time:", time.time()-start, "seconds."

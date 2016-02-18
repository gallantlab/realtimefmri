import sys
import numpy as np
import pandas as pd

log_path = sys.argv[1]
print '-'*30
print 'profiling %s\n' % log_path
def parse_log(log_path):
    def parse_line(line, split=[0,24,45,54,-1]):
        return [line[split[i]:split[i+1]].strip() for i in xrange(len(split)-1)]
    log = []
    with open(log_path, 'r') as f:
        for line in f:
            log.append(parse_line(line))
    return log

if __name__=='__main__':
    
    log = parse_log(log_path)
    log = pd.DataFrame(log, columns=['time', 'name', 'level', 'message'])
    log['time'] = pd.to_datetime(log.time, format='%Y-%m-%d %H:%M:%S,%f')
    
    starts = log.ix[log.message.str.startswith('running')]
    finishes = log.ix[log.message.str.startswith('finished')]
    starts.index = np.arange(len(starts))
    finishes.index = np.arange(len(finishes))
    
    dtime = starts.join(finishes, lsuffix='_start', rsuffix='_finish')
    dtime['d_seconds'] = (dtime.time_finish - dtime.time_start).apply(lambda x: x / np.timedelta64(1,'s'))
    dtime['step_name'] = dtime.message_start.str.replace('running ', '')
    dtime_gp = dtime.groupby('step_name').d_seconds
    
    summary = pd.DataFrame([dtime_gp.mean(), dtime_gp.min(), dtime_gp.max()]).T
    summary.columns = ['mean', 'min', 'max']
    
    print 'Summary per preprocessing step:\n'
    print summary
    print '\nAverage total time:'
    print summary.sum()
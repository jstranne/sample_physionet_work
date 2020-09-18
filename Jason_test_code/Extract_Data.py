#!/usr/bin/env python3
"""
Jason Stranne

"""
import numpy as np
import os
import sys
import scipy.io
import gc as garbageCollector
import mne

def import_signals(file_name):
    return scipy.io.loadmat(file_name)['val']

def loadSignals(recordName, dataPath):
    signals = scipy.io.loadmat(dataPath + recordName + os.sep + recordName + '.mat')
    signals = signals['val']
    garbageCollector.collect()

    return signals
def extractWholeRecord(recordName,
                       dataPath):
    #Keep all channels except ECG
    # 0 - F3-M2
    # 1 - F4-M1
    # 2 - C3-M2
    # 3 - C4-M1
    # 4 - O1-M2
    # 5 - O2-M1
    # 6 - E1-M2
    # 7 - Chin1-Chin2
    # 8 - ABD
    # 9 - CHEST
    # 10 - AIRFLOW
    # 11 - SaO2
    # 12 - ECG
    
    keepChannels = [0, 1]

    signals = loadSignals(recordName, dataPath).astype(np.float64)
    # print(signals.shape)
    # Want to add a 30Hz lowpass with hamming window
    
    signals = mne.filter.filter_data(data=signals, sfreq=200, l_freq=None, h_freq=30, method='fir', fir_window='hamming')
        
    ## 200 -> 100Hz downsample
    signals = signals[keepChannels, 0::2]

    garbageCollector.collect()

    return np.transpose(signals)


def import_sleep_stages(recordName, dataPath):
    # imports all the sleep stages as numbers in in array. A negative 1 corresponds to an undefined label.
    file_name = dataPath + recordName + os.sep + recordName + '-arousal.mat'
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')
    #undefined are -1s
    sleep_schedule = numpy.array(f['data']['sleep_stages']['undefined'])*-1 + \
        numpy.array(f['data']['sleep_stages']['nonrem3'])*0 + \
        numpy.array(f['data']['sleep_stages']['nonrem2'])*1 + \
        numpy.array(f['data']['sleep_stages']['nonrem1'])*2 + \
        numpy.array(f['data']['sleep_stages']['rem'])*3 + \
        numpy.array(f['data']['sleep_stages']['wake'])*4
    
    # downsample to 100Hz
    sleep_schedule = sleep_schedule.flatten()[0::2]
    garbageCollector.collect()
    return sleep_schedule

if __name__=="__main__":
    root = os.path.join("..","training", "")
    print(root)
    x = extractWholeRecord(recordName = "tr03-0078", dataPath = root)
    print(len(x[0]))
    y=import_sleep_stages(recordName = "tr03-0078", dataPath = root)

    print(len(y))
    print(np.max(y))
    print(np.min(y))
#!/usr/bin/env python3
"""
Jason Stranne

"""
import numpy as np
import os
import sys
import physionetchallenge2018_lib as phyc
from score2018 import Challenge2018Score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from zipfile import ZipFile, ZIP_DEFLATED
import gc


def import_signals(file_name):
    return scipy.io.loadmat(file_name)['val']

def extractWholeRecord(recordName,
                       dataPath='PATH/',
                       dataInDirectory=True):
Keep all channels except ECG
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

    signals = loadSignals(recordName, dataPath, dataInDirectory).astype(np.float64)
    print(signals.shape)
    # Want to add a 30Hz lowpass with hamming window
    signals = mne.filter.filter_data(data=signals, sfreq=200, l_freq=30, h_freq=None, method='fir', fir_window='hamming')
        
    ## 200 -> 100Hz downsample
    signals = signals[keepChannels, 0::2]

    garbageCollector.collect()

    return signals


def import_sleep_stages(file_name):
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')
    sleep_schedule = numpy.array(f['data']['sleep_stages']['undefined']) + /
        numpy.array(f['data']['sleep_stages']['nonrem3']) + /
        numpy.array(f['data']['sleep_stages']['nonrem2']) + /
        numpy.array(f['data']['sleep_stages']['nonrem1']) + /
        numpy.array(f['data']['sleep_stages']['rem']) + /
        numpy.array(f['data']['sleep_stages']['wake']) + /
    
    ##
    # Run some checks on this
    ##
    garbageCollector.collect()
    return sleep_schedule

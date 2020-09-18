#!/usr/bin/env python3
"""
Jason Stranne

"""
import numpy as np
import os
import sys
from Extract_Data import loadSignals, extractWholeRecord, import_sleep_stages
from sklearn import preprocessing
from scipy import stats


# def format_RP(recordName):
#     root = os.path.join("..","training", "")
#     x = extractWholeRecord(recordName = recordName, dataPath = root)
#     print(len(x[0]))
#     y=import_sleep_stages(recordName = recordName, dataPath = root)
#     sampling_rate = 100 # in Hz
#     window_size = 30 # in sec
    
#     print("returned shape", x.shape)
#     total_windows = len(x)//(sampling_rate*window_size)
    
#     pairs = np.random.randint(low=0, high=total_windows, size=(2000,2))
    
#     xsamples = np.zeros((2000,2,sampling_rate*window_size,2))
#     ysamples = np.zeros((2000,1))
#     for i in range(2000):
#         xsamples[i,0,:,:]=x[sampling_rate*window_size*pairs[i,0]:sampling_rate*window_size*(pairs[i,0]+1)]
#         xsamples[i,1,:,:]=x[sampling_rate*window_size*pairs[i,1]:sampling_rate*window_size*(pairs[i,1]+1)]
#         # ysamples[i]=np.median(y[sampling_rate*window_size*pairs[i,0]:sampling_rate*window_size*(pairs[i,0]+1)])
#         y_samples[i]=np.abs(sampling_rate*window_size*(pairs[i,0]-pairs[i,1]))
    
#     print(xsamples[500,1,:,:])
#     print(xsamples[500,0,:,:])
    
    
#     np.save(file=root+recordName+os.sep+recordName+"_RPdata", arr=xsamples)
#     np.save(file=root+recordName+os.sep+recordName+"_RPlabels", arr=ysamples)
    

# should also write a preprocessing file to preprocess both the x and y values (just scale down the y vals)

def preprocess_file(recordName):
    root = os.path.join("..","training", "")
    # returns 2 channels of the eeg, 30Hz hamming low pass filtered
    x = extractWholeRecord(recordName = recordName, dataPath = root)
    y = import_sleep_stages(recordName = recordName, dataPath = root)
    sampling_rate = 100 # in Hz
    window_size = 30 # in sec
    
    # print(x.shape)
    total_windows = len(x)//(sampling_rate*window_size)
    
    xwindows=[]
    sleep_labels=[]
    start_times=[]

    for i in range(total_windows):
        xval = x[sampling_rate*window_size*i:sampling_rate*window_size*(i+1)]
        yval = y[sampling_rate*window_size*i:sampling_rate*window_size*(i+1)]
        #print('x shape', xval.shape)
        #print('y shape', xval.shape)
        mode, mode_count = stats.mode(yval)
        if mode_count != sampling_rate*window_size or len(mode)>1:
            #there are conflicting signals, skip
            print("removed for bad labels")
            continue
            
        #check if we have unlabeled data, no longer removing this from RP    
#         if mode[0]<0:
#             print("removed for unlabeled")
#             continue
            
        # check if valid (peak-to-peak amplitude below 1 µV were rejected)
        if((np.max(xval[:,0])-np.min(xval[:,0])) < 1 or (np.max(xval[:,1])-np.min(xval[:,1])) < 1):
            print("removed for low signal")
            continue
        #normalized channel wize for zero mean and unit sd
        xval = preprocessing.scale(xval,axis=0)
        #xval = np.expand_dims(xval, axis=0)
        #print(xval.shape)
        #print(xwindows.shape)
        xwindows.append(xval)
        #mode is returned as a list so just get the first index
        sleep_labels.append(mode[0])
        start_times.append(window_size*i)
        
    np.save(file=root+recordName+os.sep+recordName+"_Windowed_Preprocess", arr=np.array(xwindows))
    np.save(file=root+recordName+os.sep+recordName+"_Windowed_SleepLabel", arr=np.array(sleep_labels))
    np.save(file=root+recordName+os.sep+recordName+"_Windowed_StartTime", arr=np.array(start_times))


if __name__=="__main__":
    # format_RP("tr03-0078")
    f=open(os.path.join("..","training_names.txt"),'r')
    lines = f.readlines()
    for line in lines:
        print(line.strip())
        preprocess_file(line.strip())
    f.close()
    
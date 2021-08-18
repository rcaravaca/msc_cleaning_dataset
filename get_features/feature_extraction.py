#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:53:18 2019

@author: oykukapcak
"""
import numpy as np
from sklearn.neighbors import KernelDensity 
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def extract_samples(data, start, end, sample_size, step):
    samples = []
    for i in range(start, end, step):
        sample = data[i:i+sample_size]
        samples.append(sample)
    return samples

def mutual_information(X,Y,bins):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def normalized_mutual_information(X,Y):
    bins = len(X)
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    NMI = MI/np.sqrt(H_X*H_Y)
    return NMI

#def cross_correlation(x,y, max_lag):
#    corr = plt.xcorr(x,y, maxlags=max_lag)
#    c = corr[1]
#    return c

def cross_correlation_nolag(x,y, lag=0):
    if len(x)!=len(y):
        raise('Input variables of different lengths.')

    #--------Unify types of <lag>-------------
    if np.isscalar(lag):
        if abs(lag)>=len(x):
            raise('Maximum lag equal or larger than array.')
        if lag<0:
            lag=-np.arange(abs(lag)+1)
        elif lag==0:
            lag=[0,]
        else:
            lag=np.arange(lag+1)    
    elif lag is None:
        lag=[0,]
    else:
        lag=np.asarray(lag)
    
    #-------Loop over lags---------------------
    result=[]
    for ii in lag:
        if ii<0:
            result.append(pearsonr(x[:ii],y[-ii:])[0])
        elif ii==0:
            result.append(pearsonr(x,y)[0])
        elif ii>0:
            result.append(pearsonr(x[ii:],y[:-ii])[0])
    
    result=np.asarray(result)
    
    return result

def cross_correlation_lagged(x,y, lag):
    if len(x)!=len(y):
        raise('Input variables of different lengths.')

    #--------Unify types of <lag>-------------
    if np.isscalar(lag):
        if abs(lag)>=len(x):
            raise('Maximum lag equal or larger than array.')
        if lag<0:
            lag=-np.arange(abs(lag)+1)
        elif lag==0:
            lag=[0,]
        else:
            lag=np.arange(lag+1)    
    elif lag is None:
        lag=[0,]
    else:
        lag=np.asarray(lag)
    
    #-------Loop over lags---------------------
    result=[]
    for ii in lag:
        if ii<0:
            result.append(pearsonr(x[:ii],y[-ii:])[0])
        elif ii==0:
            pass
        elif ii>0:
            result.append(pearsonr(x[ii:],y[:-ii])[0])
    
    result=np.asarray(result)
    
    return result
"""
def corr_with_significance(x,y,lag=None):
    if len(x)!=len(y):
        raise('Input variables of different lengths.')
    
    #--------Unify types of <lag>-------------
    if np.isscalar(lag):
        if abs(lag)>=len(x):
            raise('Maximum lag equal or larger than array.')
        if lag<0:
            lag=-np.arange(abs(lag)+1)
        elif lag==0:
            lag=[0,]
        else:
            lag=np.arange(lag+1)    
    elif lag is None:
        lag=[0,]
    else:
        lag=np.asarray(lag)
    
    #-------Loop over lags---------------------
    result=[]
    for ii in lag:
        if ii<0:
            result.append(pearsonr(x[:ii],y[-ii:]))
        elif ii==0:
            result.append(pearsonr(x,y))
        elif ii>0:
            result.append(pearsonr(x[ii:],y[:-ii]))
    
    result=np.asarray(result)
    
    return result
    
        
def corr_with_p(x,y):
    return pearsonr(x,y)

def corrlagged_with_p(x,y, lag):
    if len(x)!=len(y):
        raise('Input variables of different lengths.')
    
    #--------Unify types of <lag>-------------
    if np.isscalar(lag):
        if abs(lag)>=len(x):
            raise('Maximum lag equal or larger than array.')
        if lag<0:
            lag=-np.arange(abs(lag)+1)
        elif lag==0:
            lag=[0,]
        else:
            lag=np.arange(lag+1)    
    elif lag is None:
        lag=[0,]
    else:
        lag=np.asarray(lag)
    
    #-------Loop over lags---------------------
    result=[]
    for ii in lag:
        if ii<0:
            result.append(pearsonr(x[:ii],y[-ii:]))
        elif ii==0:
            pass
        elif ii>0:
            result.append(pearsonr(x[ii:],y[:-ii]))
    
    #result=np.asarray(result)
    
    return result
"""
"""
training is 30 seconds (female) test is consecutive 5 seconds (male)
what it checks: does person mimics the behavior of interlocutor? compares 5 sec behavior to 30 secs
inputs are in frames, so if they are seconds just multiply with 20 beforehand
for complex features such as psd or var just input number?
"""
"""
def mimicry(data_train, data_test, sample_size_train=600, step_size_train=100, sample_size_test=100, step_size_test=100, start_test=600, method="kde"):
    #first extract samples for training
    sz = len(data_train)
    samples_train = []

    # Use the sliding window of sample size seconds with step size shift
    start_train = 0
    end_train = sz-sample_size_train
    samples_train = extract_samples(data_train, start_train, end_train, sample_size_train, step_size_train)
    #for i in range(0,sz-sampleSize,stepSize):
    #    sample = data_train[i:i+sampleSize]
    #    samples_train.append(sample)

    #then extract samples for test
    sz = len(data_test)
    samples_test = []
    
    end_test = sz
    # Use the sliding window of sample size seconds with step size shift 
    # skip first 30 seconds or whatever the input is
    samples_test = extract_samples(data_test, start_test, end_test, sample_size_test, step_size_test)
    #for i in range(start,sz,stepSize):
    #    sample = data_test[i:i+sampleSize]
    #    samples_test.append(sample)
 

    scores = []
    for (trSample, teSample) in zip(samples_train, samples_test):
        #kde = sm.nonparametric.KDEUnivariate(trSample)
        #kde.fit()
        #eval_kde =kde.evaluate([teSample])
        #score = np.mean(eval_kde)
        
        #compute log likelihood for each training sample for the corresponding test sample
        #then take mean to obtain one single score
        #final "scores" is the list of all samples log likelihoods
        if method == 'kde':     
            score = np.mean(kde_sklearn(trSample, teSample))
        else:
            sum_train = np.sum(trSample, axis=0)
            div_train = np.divide(sum_train, len(trSample))
            sum_test = np.sum(teSample, axis=0)
            div_test = np.divide(sum_test, len(teSample))
            #score = cosine_similarity([div_test], [div_train])
            if method == 'cos':
                score = 1- cosine(div_test, div_train) 
            elif method == 'ssd':
                score = np.sum((div_train-div_test)**2) 
    
        scores.append(score)
    
    #can compute min, max, mean, std from these scores
    std = np.std(scores)
    mean = np.mean(scores)
    minimum = np.amin(scores)
    maximum = np.amax(scores)
    result = [std, mean, minimum, maximum]
    return result
"""
def mimicry_new(data_train, data_test):
    scores = []
    for i in range(0, len(data_train)-1):
        trSample = data_train[i]
        teSample = data_test[i+1]
        score = np.sum((trSample-teSample)**2) 
        scores.append(score)
    
    #can compute min, max, mean, std from these scores
    std = np.std(scores)
    mean = np.mean(scores)
    minimum = np.amin(scores)
    maximum = np.amax(scores)
    result = [std, mean, minimum, maximum]
    return result
        
"""
training: first 2 mins
test: last 1 min divided into 5 secs
compare 5 sec segments to training
obtain 12 scores
then compute pearson correlation over time
couldn't figure out how to compute pearson correlation though
"""
"""
def conv1(data_train, data_test, sample_size_train=2400, sample_size_test =100, step_size_test=100, start_test=2400, start_train=0, method="kde"):
    train_2min = data_train[start_train:start_train+sample_size_train]
    
    sz = len(data_test)
    end_test = sz
    # Use the sliding window of sample size seconds with step size shift 
    # skip first 30 seconds or whatever the input is
    samples_test = extract_samples(data_test, start_test, end_test, sample_size_test, step_size_test)
    
    scores = []
    if method == "kde":
        for sample in samples_test:
            score = np.mean(kde_sklearn(train_2min, sample))
            scores.append(score)

    else:
        sum_train = np.sum(train_2min, axis=0)
        div_train = np.divide(sum_train, len(train_2min))  #obtains one psd value for whole 2 minutes of training

        scores = []
        for sample in samples_test:
            sum_test = np.sum(sample, axis=0)
            div_test = np.divide(sum_test, len(sample))
            if method == 'cos':
                #score = cosine_similarity([div_test], [div_train])
                score = 1- cosine(div_test, div_train)
            elif method == 'ssd':
                score = np.sum((div_train-div_test)**2) 
            scores.append(score)               
    
    #here we need to find a way to correlate time with scores and return that value!!!!! 
    corr = 0 
    
    time = np.arange(len(scores))
    corr = np.corrcoef(scores,time)[0][1]

    
    corr = plt.xcorr(time, scores)
    lags = corr[0]
    c = corr[1]
    
    np.correlate(time,scores,"full")
    
    pearsonr(time, scores)


    return corr
"""
def conv1_new(data_train, data_test):
    train_end = int(len(data_train)/3*2)
    train_2min = data_train[0:train_end]
    
    # Use the sliding window of sample size seconds with step size shift 
    # skip first 30 seconds or whatever the input is
    test_samples = []
    for i in range(train_end, len(data_test)-1):
        sample = data_test[i]
        test_samples.append(sample)  
    
    sum_train = np.sum(train_2min, axis=0)
    div_train = np.divide(sum_train, len(train_2min))  #obtains one psd value for whole 2 minutes of training
    
    scores = []
    for sample in test_samples:
        score = np.sum((div_train-sample)**2) 
        scores.append(score)               
    
    #here we need to find a way to correlate time with scores and return that value!!!!! 
    time = np.arange(len(scores))
    corr =  pearsonr(time, scores)[0] 
    return corr

"""  
def conv1_with_significance(data_train, data_test):
    train_end = int(len(data_train)/3*2)
    train_2min = data_train[0:train_end]
    
    # Use the sliding window of sample size seconds with step size shift 
    # skip first 30 seconds or whatever the input is
    test_samples = []
    for i in range(train_end, len(data_test)-1):
        sample = data_test[i]
        test_samples.append(sample)  
    
    sum_train = np.sum(train_2min, axis=0)
    div_train = np.divide(sum_train, len(train_2min))  #obtains one psd value for whole 2 minutes of training
    
    scores = []
    for sample in test_samples:
        score = np.sum((div_train-sample)**2) 
        scores.append(score)               
    
    #here we need to find a way to correlate time with scores and return that value!!!!! 
    time = np.arange(len(scores))
    return(pearsonr(time,scores))
"""
"""

def conv2(data_train, data_test, sample_size_train=2400, sample_size_test=1200, start_train=0, start_test_first=0, start_test_last=2400, method="kde"):
    train_2min = data_train[start_train:start_train+sample_size_train]

    test_1min_first = data_test[start_test_first:start_test_first+sample_size_test]
    test_1min_last = data_test[start_test_last:start_test_last+sample_size_test]
    
    if method == "kde":
        score_first = np.mean(kde_sklearn(train_2min, test_1min_first))
        score_last = np.mean(kde_sklearn(train_2min, test_1min_last))
    
    else:
        sum_train = np.sum(train_2min, axis=0)
        div_train = np.divide(sum_train, len(train_2min))
        sum_test1 = np.sum(test_1min_first, axis=0)
        div_test1 = np.divide(sum_test1, len(test_1min_first))
        sum_test2 = np.sum( test_1min_last, axis=0)
        div_test2 = np.divide(sum_test2, len(test_1min_last))
        if method == 'cos':
            #score = cosine_similarity([div_test], [div_train])
            score_first = 1- cosine(div_test1, div_train)
            score_last = 1- cosine(div_test2, div_train)
        elif method == 'ssd':
            score_first = np.sum((div_train-div_test1)**2) 
            score_last = np.sum((div_train-div_test2)**2)             
    score = score_last - score_first
    return score
"""

#training: first 2 minutes of participant1
#test: first and last 1 minutes of participant2
#both are evaluated and final score is computed by subtracting first_score from last_score
#positive result means "last" fits better meaning convergence
"""
def conv2_new(data_train, data_test):
    train_end = int(len(data_train)/3*2)
    train_2min = data_train[0:train_end]
    
    sample_size_test = int(len(data_test)/3)
    test_1min_first = data_test[0:sample_size_test]
    test_1min_last = data_test[train_end:train_end+sample_size_test]
    
    sum_train = np.sum(train_2min, axis=0)
    div_train = np.divide(sum_train, len(train_2min))
    sum_test1 = np.sum(test_1min_first, axis=0)
    div_test1 = np.divide(sum_test1, len(test_1min_first))
    sum_test2 = np.sum( test_1min_last, axis=0)
    div_test2 = np.divide(sum_test2, len(test_1min_last))
    
    score_first = np.sum((div_train-div_test1)**2) 
    score_last = np.sum((div_train-div_test2)**2)      
    
    score = score_last - score_first
    return score
"""
#def conv2_new2(data_train, data_test):
#    first_third_end = int(len(data_train)/3*1)
#    train_first = data_train[0:first_third_end]
#    test_first = data_test[0:first_third_end]
#
#    last_third_start = int(len(data_train)/3*2)
#    last_third_end = len(data_train)
#    train_last = data_train[last_third_start:last_third_end]
#    test_last = data_test[last_third_start:last_third_end]    
#        
#
#    sum_train_first = np.sum(train_first, axis=0)
#    div_train_first = np.divide(sum_train_first, len(train_first))
#    sum_test_first = np.sum(test_first, axis=0)
#    div_test_first = np.divide(sum_test_first, len(test_first))
#
#    sum_train_last = np.sum(train_last, axis=0)
#    div_train_last = np.divide(sum_train_last, len(train_last))
#    sum_test_last = np.sum(test_last, axis=0)
#    div_test_last = np.divide(sum_test_last, len(test_last))
#
#    score_first = np.sum((div_train_first-div_test_first)**2) 
#    score_last = np.sum((div_train_last-div_test_last)**2)      
#    
#    score = score_last - score_first
#    return score

def conv2_new2(data_train, data_test):
    first_half_end = int(len(data_train)/2)
    train_first = data_train[0:first_half_end]
    test_first = data_test[0:first_half_end]

    last_half_start = int(len(data_train)/3*2)
    last_half_end = len(data_train)
    train_last = data_train[last_half_start:last_half_end]
    test_last = data_test[last_half_start:last_half_end]    
        

    sum_train_first = np.sum(train_first, axis=0)
    div_train_first = np.divide(sum_train_first, len(train_first))
    sum_test_first = np.sum(test_first, axis=0)
    div_test_first = np.divide(sum_test_first, len(test_first))

    sum_train_last = np.sum(train_last, axis=0)
    div_train_last = np.divide(sum_train_last, len(train_last))
    sum_test_last = np.sum(test_last, axis=0)
    div_test_last = np.divide(sum_test_last, len(test_last))

    score_first = np.sum((div_train_first-div_test_first)**2) 
    score_last = np.sum((div_train_last-div_test_last)**2)      
    
    score = score_last - score_first
    return score


#this one computes the similarity between psd's 
"""
def conv3(data_train, data_test, sample_size_train=3, step_size_train=3, sample_size_test =3, step_size_test=3, start_test=0, start_train=0, method="ssd"):
    sz = len(data_train)
    end_train = sz-step_size_train
    samples_train = extract_samples(data_train, start_train, end_train, sample_size_train, step_size_train)
    
    sz = len(data_test)
    end_test = sz-step_size_test
    samples_test = extract_samples(data_test, start_test, end_test, sample_size_test, step_size_test)

    scores = []
    for (trSample, teSample) in zip(samples_train, samples_test):
        #compute log likelihood for each training sample for the corresponding test sample
        #then take mean to obtain one single score
        #final "scores" is the list of all samples log likelihoods
        if method == 'kde':     
            score = np.mean(kde_sklearn(trSample, teSample))
        else:
            sum_train = np.sum(trSample, axis=0)
            div_train = np.divide(sum_train, len(trSample))
            sum_test = np.sum(teSample, axis=0)
            div_test = np.divide(sum_test, len(teSample))
            #score = cosine_similarity([div_test], [div_train])
            if method == 'cos':
                score = 1- cosine(div_test, div_train) 
            elif method == 'ssd':
                score = np.sum((div_train-div_test)**2) 
        scores.append(score)
    
    #need to compute the correlation of scores with time to see if they increase or not    
    time = np.arange(len(scores))
    corr = np.corrcoef(scores,time)[0][1]
    return corr
"""
def conv3_new(data_train, data_test):
    scores = []
    for i in range(0, len(data_train)-1):
        trSample = data_train[i]
        teSample = data_test[i]
        score = np.sum((trSample-teSample)**2) 
        scores.append(score)
    
    #need to compute the correlation of scores with time to see if they increase or not    
    time = np.arange(len(scores))
    corr =  pearsonr(time, scores)[0] 
    return corr

    
"""
def conv3_with_significance(data_train, data_test):
    scores = []
    for i in range(0, len(data_train)-1):
        trSample = data_train[i]
        teSample = data_test[i]
        score = np.sum((trSample-teSample)**2) 
        scores.append(score)
    
    #need to compute the correlation of scores with time to see if they increase or not    
    time = np.arange(len(scores))
    return(pearsonr(time,scores))
"""
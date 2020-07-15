#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:38:11 2020

@author: peter
"""

import numpy as np
import FRI_functions as FRIF
import matplotlib.pyplot as plt
import scipy.stats
import cosmic.cosmic
import extract_exponentials as ee
#np.random.seed(0)

def detect_spikes(jhist,bins,all_tk,all_ak,thresh):
    peaks = scipy.signal.find_peaks(jhist/np.nanmax(jhist) > thresh)[0]
    sp_t = bins[peaks]  + np.mean(np.diff(bins))/2
    delta = np.mean(np.diff(bins))
    precision = 1
    amplitudes = np.zeros(len(sp_t))
    for idx,ti in enumerate(sp_t):
        sp_tk = np.argwhere(np.abs(all_tk[0] - ti) < precision*delta)
        sp_ak = np.mean(all_ak[0][sp_tk.ravel()])
        amplitudes[idx] = sp_ak
        
    return sp_t,amplitudes

def double_consistency_histogram(x,t,tau,winlens = [32,8],
                                 fixed_K = [None,1],
                                 spike_thresh = 0.1,
                                 hist_res = 1,
                                 hist_thresh = 0.05,
                                 shutter_length = None):
    
    all_tk = []
    all_ak = []

    for idx,win_len in enumerate(winlens):
        if shutter_length is None:
            tk,ak = ee.sliding_window_detect(t,x,tau,win_len,fixed_K = fixed_K[idx])
        else:
            tk,ak = ee.sliding_window_detect_box_filtered(t, x, tau, win_len, shutter_length,fixed_K = fixed_K[idx])
        
        all_tk.append(np.concatenate(tk))
        all_ak.append(np.concatenate(ak))
        
        
    #remove spikes below certain size?
    for idx in range(len(all_ak)):
        keep = all_ak[idx] > spike_thresh
        all_ak[idx] = all_ak[idx][keep]
        all_tk[idx] = all_tk[idx][keep]

    
    #generate histogram
    bins = np.linspace(t[1],t[-1],int(len(t)*hist_res))
    
    all_hists = []
    
    for idx,tk in enumerate(all_tk):
        hist,_ = np.histogram(tk,bins = bins,density = True)
        all_hists.append(hist)
    
    all_hists = np.array(all_hists)
    
    jhist = all_hists[0]*all_hists[1]
    
    tk,ak = detect_spikes(jhist, bins, all_tk, all_ak, hist_thresh)


    return tk,ak,(jhist,bins,all_tk,all_ak)


def compare_spike_trains(tk,tk_true,noise_level,fs,tau):
    #use cosmic metric for spike train comparison
    crb       = cosmic.cosmic.compute_crb(1/fs, 1, noise_level**2, alpha = 1/tau, gamma = 10**3)
    # get metric width from crb
    width     = cosmic.cosmic.compute_metric_width(crb)

    cos_score, cos_prec, cos_call,y,y_hat,t  = cosmic.cosmic.compute_score(width,tk_true,tk)
    return cos_score,(cos_prec, cos_call,y,y_hat,t)

def calculate_ROC(length,noise_level,evals = 25):
    np.random.seed(0)
    fs = 8
    tau = 0.5
    num_spikes = 0
    while num_spikes == 0:
        tk_true,ak_true,t,x = FRIF.make_signal(length,fs,tau = tau, noise_level = noise_level,spike_std = 0)
        num_spikes = len(tk_true)
    
    #use cosmic metric for spike train comparison
    crb       = cosmic.cosmic.compute_crb(1/fs, np.mean(ak_true), noise_level**2, 1/tau, 10**3)
    # get metric width from crb
    width     = cosmic.cosmic.compute_metric_width(crb)
    
    _,_,jhist_ret = double_consistency_histogram(x,t,tau)
    jhist,bins,all_tk,all_ak = jhist_ret
    
    scores = []
    prec = []
    call = []
    
    for thresh_val in np.arange(evals)/evals:
        tk,_ = detect_spikes(jhist, bins, all_tk, all_ak, thresh_val)
        print(len(tk))
        cos_score, cos_prec, cos_call,_,_, _ = cosmic.cosmic.compute_score(width,tk_true,tk)
        scores.append(cos_score)
        prec.append(cos_prec)
        call.append(cos_call)
        
    
    fig,ax = plt.subplots()
    ax.plot(call,prec)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    return scores,prec,call



def example(length,noise_level):
    fs = 30
    tau = 0.5
    num_spikes = 0
    while num_spikes == 0:
        tk_true,ak_true,t,x = FRIF.make_signal(length,fs,tau = tau, noise_level = noise_level,spike_std = 0)
        num_spikes = len(tk_true)

    tk,ak,_ = double_consistency_histogram(x, t, tau,hist_thresh = 0.1,winlens = [64,16],spike_thresh = 0.05)
    
    plt.cla()
    plt.plot(t,x)
    plt.plot(t,x,'.')
    plt.stem(tk_true,ak_true,'r', label = 'True',markerfmt='ro')
    plt.stem(tk,ak,'k',label = 'Detected',markerfmt ='xk',linefmt = '--k')
    plt.legend()
    return tk,ak
    
if __name__ == '__main__':
    tk,ak = example(40,0.1)
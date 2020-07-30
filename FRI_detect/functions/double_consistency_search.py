#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:38:11 2020

@author: peter
"""

import numpy as np
import scipy.stats
import cosmic.cosmic


from FRI_detect.functions import extract_exponentials as ee


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
    '''
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    winlens : TYPE, optional
        DESCRIPTION. The default is [32,8].
    fixed_K : TYPE, optional
        DESCRIPTION. The default is [None,1].
    spike_thresh : TYPE, optional
        DESCRIPTION. The default is 0.1.
    hist_res : TYPE, optional
        DESCRIPTION. The default is 1.
    hist_thresh : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    tk : TYPE
        DESCRIPTION.
    ak : TYPE
        DESCRIPTION.
    hist_data : TYPE
        DESCRIPTION.

    '''
    
    all_tk = []
    all_ak = []

    for idx,win_len in enumerate(winlens):
        if shutter_length is None:
            tk,ak = ee.sliding_window_detect(t,x,tau,win_len,fixed_K = fixed_K[idx])
        else:
            tk,ak = ee.sliding_window_detect_box_filtered(t,x,tau,win_len,shutter_length,fixed_K = fixed_K[idx], taper_window = True)
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

    hist_data = (jhist,bins,all_tk,all_ak)
    return tk,ak,hist_data


def compare_spike_trains(tk,tk_true,noise_level,fs,tau):
    #use cosmic metric for spike train comparison
    crb       = cosmic.cosmic.compute_crb(1/fs, 1, noise_level**2, alpha = 1/tau, gamma = 10**3)
    # get metric width from crb
    width     = cosmic.cosmic.compute_metric_width(crb)

    cos_score, cos_prec, cos_call,y,y_hat,t  = cosmic.cosmic.compute_score(width,tk_true,tk)
    return cos_score,(cos_prec, cos_call,y,y_hat,t)


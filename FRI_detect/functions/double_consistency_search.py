#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:38:11 2020

@author: peter
"""

import numpy as np
import scipy.stats

try: 
    import cosmic.cosmic
except ImportError: 
    cosmic = None

from FRI_detect.functions import extract_exponentials as ee


def detect_spikes(jhist,bins,all_tk,all_ak,thresh):
    '''
    Finds spikes from a double consistency histogram of spike times

    '''
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
    Extracts spike times from the fluorescent time course x, which contains 
    calcium transients with instantaneous rise and decay constant tau.
    Uses a double consitency method as in https://doi.org/10.25560/49792 sec 4

    Parameters
    ----------
    x : 1D float vec
        The fluorescent signal.
    t : 1D float vec
        Time stamps of x.
    tau : float
        Exponential decay const, e^(-t/tau).
    winlens : vec of ints length 2, optional
        The window lengths to use for double consistency search. The default is [32,8].
    fixed_K : vec of None or ints, length 2, optional
        Wether to estimate (None) or assume specific number of spikes per window. The default is [None,1].
        Order dictates which window length they refer to
    spike_thresh : Float, optional
        Spike size below which to reject spike detections. The default is 0.1.
    hist_res : float, optional
        Factor to oversample the histogram sampling. The default is 1.
    hist_thresh : float, optional
        Threshold to remove noise from histogram. The default is 0.05.
    shutter_length : float, optional
        If is None then assumes an integrating detector (i.e. camera) has filtered the 
        fluorescence time course with period given by this. The default is None.

    Returns
    -------
    tk : 1d vector of float
        detected spike times.
    ak : 1d vec of float
        detected spike amplitudes.
    hist_data : tuple
        Histogram and all spikes found from windows for debugging.

    '''
    
    all_tk = []
    all_ak = []

    #Run spike extraction in sliding window for different win lengths
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

    
    #generate histogram of all the spikes
    bins = np.linspace(t[1],t[-1],int(len(t)*hist_res))
    
    all_hists = []
    
    for idx,tk in enumerate(all_tk):
        hist,_ = np.histogram(tk,bins = bins,density = True)
        all_hists.append(hist)
    
    all_hists = np.array(all_hists)
    
    #multipl together to weight both
    jhist = all_hists[0]*all_hists[1]
    
    #get spikes from histogram peaks
    tk,ak = detect_spikes(jhist, bins, all_tk, all_ak, hist_thresh)

    hist_data = (jhist,bins,all_tk,all_ak)
    return tk,ak,hist_data


def extract_times(all_tk, T, win_len):
    '''
    A function to extract spike times from a single pass of the sliding window 
    by looking at similarity between adjacent windows.

    Parameters
    ----------
    all_tk : list of 1d vecs
        List of spike times detected in each window.
    T : float
        sampling period.
    win_len : even int
        Window length used to extract spikes.

    Returns
    -------
    1d vec of floats
        Spike times.

    '''
    #Look for spikes constently detected within 1.5 sampling periods and use
    #the mean detected time as the spike time
    all_tk = np.sort(all_tk)
    interspike = (np.diff(all_tk) > T/1.5).astype(int)
    sta = np.squeeze(np.argwhere(np.diff(interspike) < 0))
    stop = np.squeeze(np.argwhere(np.diff(interspike) > 0))
    if interspike[0] == 0:
        sta = np.concatenate(([0],sta))
    if interspike[-1] == 0:
        stop = np.concatenate((stop,[len(all_tk)]))
    
    tk = []
    for idx in range(len(sta)):
        if stop[idx] - sta[idx] > win_len/5:
            tk.append(np.mean(all_tk[sta[idx]:stop[idx]]))
        
    return np.array(tk)

def single_search(x,t,tau,winlen,shutter_length = None):
    '''
    Extracts spike times from the fluorescent time course x, which contains 
    calcium transients with instantaneous rise and decay constant tau.
    uses a single pass search and extracts true spikes from adjacent windows extracting
    spikes in the same position.

    Parameters
    ----------
     x : 1D float vec
        The fluorescent signal.
    t : 1D float vec
        Time stamps of x.
    tau : float
        Exponential decay const, e^(-t/tau).
    winlen : int
        Window length to use.
    shutter_length : float, optional
        If is None then assumes an integrating detector (i.e. camera) has filtered the 
        fluorescence time course with period given by this. The default is None.

    Returns
    -------
    tk : 1d vec of float
        Spike times detected.

    '''
    t_max = t.max()
    t_min = t.min()
    #first pad beginning and end of x to size of win len to make sure spikes at beginnign and end detected
    T= np.mean(np.diff(t))
    x = np.pad(x,(winlen,winlen),mode = 'edge')
    t = np.linspace(-1*winlen*T,t[-1]+T*winlen,len(x))
    
    #extract sliding window
    if shutter_length is None:
        all_tk,_ = ee.sliding_window_detect(t,x,tau,winlen,fixed_K = None)
    else:
        all_tk,_ = ee.sliding_window_detect(t,x,tau,winlen,fixed_K = None,taper_window=True)
        
    #keep spikes that are consecutively detected
    tk = extract_times(np.concatenate(all_tk),np.mean(np.diff(t)),winlen)
    
    tk = tk[(tk <t_max)&(tk > t_min)]
    return tk

 
def compare_spike_trains(tk,tk_true,noise_level,fs,tau):
    '''
    Uses the CosMIC metric (https://github.com/stephanierey/metric) to score
    the accuracy of an estimated spike train
    Basically encapsulates the above code.

    Parameters
    ----------
    tk : 1d vec of floats
        The estimated spike times.
    tk_true : 1D vec of floats
        The true spike times.
    noise_level : float
        The std of the fluorescence time course.
    fs : float
        sampling rate.
    tau : float
        exponential decay time of the indicator (e^(-t/tau)).

    Returns
    -------
    float
        the similarity score (0 -> 1, higher better).
    tuple
        The rest of the return from the CosMIC code.

    '''
    if cosmic is None:
        raise ImportError('This function requires CosMIC installed as package from https://github.com/stephanierey/metric')
        
        
    #use cosmic metric for spike train comparison
    crb       = cosmic.cosmic.compute_crb(1/fs, 1, noise_level**2, alpha = 1/tau, gamma = 10**3)
    # get metric width from crb
    width     = cosmic.cosmic.compute_metric_width(crb)

    #Cosmic raises an error if there are no spikes
    if len(tk)!= 0:
        cos_score, cos_prec, cos_call,y,y_hat,t  = cosmic.cosmic.compute_score(width,tk_true,tk)
        return cos_score,(cos_prec, cos_call,y,y_hat,t)
    else:
        return 0,0


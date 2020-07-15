#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:36:20 2020

@author: peter
"""

import functions.double_consistency_search as dc
import numpy as np


def detect_spikes(x,t,tau, windows = []):
    '''
    Parameters
    ----------
    x : 1D array of floats
        Fluorescent signal consisting of noisy measurements of instantaneous rise
        calcium transients.
    t : 1D array of floats
        Time stamps of the signal in seconds.
    tau : float
        decay constant of the exponentials in seconds.
    windows : 1D array of even integers, optional
        The window lengths for the sliding window detection. The default is [].
        Will be estimated if not given.

    Returns
    -------
    tk : 1D array of floats
        Detected spike times.

    '''
    
    
    if windows == []:
        T = np.mean(np.diff(t))
        
        win_len1 = np.round(T*32*8)
        if win_len1%2 != 0:
            win_len1 -= 1
            
        win_len2 = win_len1/2
        
        windows = [win_len1,win_len2]

    
    tk,_,_ = dc.double_consistency_histogram(x,t,tau,winlens = windows,
                                 fixed_K = [None,1],
                                 spike_thresh = 0.1,
                                 hist_res = 1,
                                 hist_thresh = 0.05)
    return tk
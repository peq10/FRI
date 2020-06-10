#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:38:11 2020

@author: peter
"""

import numpy as np
import ca_detect_sliding_emom as ca_detect
import FRI_functions as FRIF
import matplotlib.pyplot as plt
import scipy.stats
#np.random.seed(0)

def detect_spikes(jhist,bins,all_tk,all_ak,thresh,):
    peaks = scipy.signal.find_peaks(jhist/np.nanmax(jhist) > thresh)[0]
    sp_t = bins[peaks]
    delta = np.mean(np.diff(bins))
    precision = 1
    amplitudes = np.zeros(len(sp_t))
    for idx,ti in enumerate(sp_t):
        sp_tk = np.argwhere(np.abs(all_tk[0] - ti) < precision*delta)
        sp_ak = np.mean(all_ak[0][sp_tk.ravel()])
        amplitudes[idx] = sp_ak
        
    return sp_t,amplitudes

def double_consistency_histogram(x,t,tau,winlens = [32,8],
                                 modes = ['estimate','fixed'],
                                 fixed_K = 1, 
                                 spike_thresh = 0,
                                 hist_res = 1,
                                 hist_thresh = 0.05):
    
    all_tk = []
    all_ak = []

    for idx,win_len in enumerate(winlens):
        if modes[idx] == 'fixed':
            fixed_K_val= fixed_K
        else:
            fixed_K_val = None
                
        tk,ak,_,_ = ca_detect.sliding_window_detect(x, t, win_len, tau, mode = modes[idx], fixed_K = fixed_K_val)
        all_tk.append(tk)
        all_ak.append(ak)
        
        
    #remove spikes below certain size?
    for idx in range(len(all_ak)):
        keep = all_ak[idx] > spike_thresh*np.max(all_ak[idx])
        all_ak[idx] = all_ak[idx][keep]
        all_tk[idx] = all_tk[idx][keep]

    
    #generate histogram
    hist_res = 1
    bins = np.linspace(t[1],t[-1],int(len(t)*hist_res))
    
    all_hists = []
    
    for idx,tk in enumerate(all_tk):
        hist,_ = np.histogram(tk,bins = bins,density = True)
        all_hists.append(hist)
    
    all_hists = np.array(all_hists)
    #fig,ax = plt.subplots()
    #ax.plot(bins[:-1],all_hists[0]/all_hists[0].max())
    #ax.plot(bins[:-1],all_hists[1]/all_hists[1].max())
    
    jhist = all_hists[0]*all_hists[1]
    
    tk,ak = detect_spikes(jhist, bins, all_tk, all_ak, hist_thresh)


    return tk,ak



def example(length,noise_level):
    fs = 8
    tau = 0.5
    num_spikes = 0
    while num_spikes == 0:
        tk_true,ak_true,t,x = FRIF.make_signal(length,fs,tau = tau, noise_level = noise_level)
        num_spikes = len(tk_true)

    tk,ak = double_consistency_histogram(x, t, tau)
    
    fig,ax = plt.subplots()
    ax.plot(t,x)
    ax.stem(tk,ak,'k',label = 'Detected',markerfmt ='xk')
    ax.stem(tk_true,ak_true,'r', label = 'True',markerfmt='ro')
    plt.legend()
    
if __name__ == '__main__':
    example(20,0.2)
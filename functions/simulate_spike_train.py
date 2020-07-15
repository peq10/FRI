#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:40:31 2020

@author: peter
"""
import numpy as np

def draw_poisson_times(window_length,firing_rate, spike_size = 1, spike_std = 0.25):
    #follow onativia method
    #divide into bins of ~ 1 firing period
    nbins = np.round(window_length*firing_rate).astype(int)
    bin_length = window_length/nbins
    short_firing = firing_rate*window_length/nbins
    
    #draw how many spikes
    num_spikes = np.random.poisson(lam = short_firing,size = nbins)
    
    #draw exact spike time from uniform distribution within bin
    tk = []
    for idx in range(nbins):
        n = num_spikes[idx]
        tk += list(np.random.uniform(size = n)*bin_length + idx*bin_length)
    tk = np.array(tk)
    #draw from lognormal distribution for spike size
    ak = np.random.lognormal(mean = spike_size,sigma = spike_std,size = len(tk))
    #print(num_spikes)
    return tk,ak

def make_signal(length,fs,firing_rate = 0.5,tau = 0.5,spike_size = 1, spike_std = 0.25, noise_level = 0):
    t = np.arange(0,length,1/fs)
    
    #draw times so that always 'fully contained' within time course
    tk,ak = draw_poisson_times(length - 4*tau,firing_rate,spike_size = spike_size,spike_std = spike_std)
    
    signal = np.zeros_like(t)
    for idx,sp in enumerate(tk):
        t_adj = t - sp
        t_adj *= t_adj > 0
        signal += (t >= sp)*np.exp(-t_adj/tau)*ak[idx]
        
    if noise_level > 0:
        signal += np.random.normal(loc = 0, scale = noise_level*spike_size,size = len(signal))
        
    return tk,ak,t,signal

    
    
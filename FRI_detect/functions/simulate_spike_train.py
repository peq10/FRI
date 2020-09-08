#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:40:31 2020

@author: peter
"""
import numpy as np

def draw_poisson_times(window_length,firing_rate, spike_size = 1, spike_std = 0.25):
    '''
    Using similar method to https://doi.org/10.25560/49792 sec 4.2
    spike amplitudes are drawn from a lognormal distribution    
    
    Parameters
    ----------
    window_length : float
        0 -> this are valid times.
    firing_rate : float
        Poisson rate of spiking.
    spike_size : float, optional
        Spike size lognormal mean. The default is 1.
    spike_std : float, optional
        Spike size lognormal standard. The default is 0.25.

    Returns
    -------
    tk : TYPE
        DESCRIPTION.
    ak : TYPE
        DESCRIPTION.

    '''

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
    '''
    Makes a fluorescent signal with random poisson spike times and 
    lognormally distributed amplitudes

    Parameters
    ----------
     length : Float
        length of signal.
    fs : float
        sampling length.
    firing_rate : float, optional
        Poisson expected rate. The default is 0.5.
    tau : float, optional
        decay constant of transients. The default is 0.5.
    spike_size : float, optional
        mean of lognormal amplitude dist. The default is 1.
    spike_std : float, optional
        Std of lognormal spike amplitdue dist. The default is 0.25.
    noise_level : float, optional
        Std of additive gaussian noise . The default is 0.

    Returns
    -------
    tk : 1d vec of float
        spike times.
    ak : 1d vec of float
        spike amplitudes.
    t :  1d vec of floats
        signal time stamps.
    signal : 1d vec of floats
        Fluorescent signal.

    '''
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


def make_signal_deterministic(length,fs,tk,ak,tau,noise_level = 0):
    '''
    Makes a signal with specified spike times and sizes.

    Parameters
    ----------
    length : Float
        length of signal.
    fs : float
        sampling length.
    tk : 1d vector of floats
        spike times.
    ak : 1d vector of floats - same size as tk
        spike amplitudes.
    tau : float
        exponential decay constant e^(-t/tau).
    noise_level : float, optional
        Standard deviation of signal noise level. The default is 0.

    Returns
    -------
    t : 1d array of floats
        time stamps.
    signal : 1d array of floats
        Fluorescent signal.

    '''
    t = np.arange(0,length,1/fs)
    
    
    signal = np.zeros_like(t)
    for idx,sp in enumerate(tk):
        t_adj = t - sp
        t_adj *= t_adj > 0
        signal += (t >= sp)*np.exp(-t_adj/tau)*ak[idx]
        
    if noise_level > 0:
        signal += np.random.normal(loc = 0, scale = noise_level,size = len(signal))
        
    return t,signal
    
    
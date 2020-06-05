#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:40:31 2020

@author: peter
"""
import numpy as np
import scipy.signal
import generate_e_spline as ges
import get_c_mn_exp as gcm
import matrix_pencil as mp



def generate_e_spline(alpha_vec,T_s,T=1, mode = 'causal'):
    return ges.generate_e_spline(alpha_vec,T_s,T=T, mode = mode)

def get_c_mn_exp(alpha_m,n,phi,t_phi):
    return gcm.get_c_mn_exp(alpha_m,n,phi,t_phi)

def get_c_mn_exp2(alpha_m,n,phi,t_phi):
    return gcm.get_c_mn_exp2(alpha_m,n,phi,t_phi)


def acmp_p(sm, K, M, P, p):
    return mp.acmp_p(sm, K, M, P, p)

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
    print(num_spikes)
    return tk,ak

def make_signal(length,fs,firing_rate = 0.5,tau = 0.5,spike_size = 1, spike_std = 0.25):
    t = np.arange(0,length,1/fs)
    
    #draw times so that always 'fully contained' within time course
    tk,ak = draw_poisson_times(length - 4*tau,firing_rate,spike_size = spike_size,spike_std = spike_std)
    
    signal = np.zeros_like(t)
    for idx,sp in enumerate(tk):
        t_adj = t - sp
        t_adj *= t_adj > 0
        signal += (t >= sp)*np.exp(-t_adj/tau)*ak[idx]
    return tk,ak,t,signal

def make_delta_signal(length,fs,kernel,rate,spike_size = 1, spike_std = 0.25,over_samp = 512):
    '''
    Simulates signal of delta functions as sampled by kernel
    Kernel discretised at fs*over_samp
    '''
    t = np.arange(0,length,1/fs/over_samp)
    
    kernel_length = np.sum(kernel > 0)/fs/over_samp
    #draw times so that always 'fully contained' within time course
    tk,ak = draw_poisson_times(length - kernel_length,rate,spike_size = spike_size,spike_std = spike_std)
    
    #make delta function
    x = np.zeros_like(t)
    for idx, sp in enumerate(tk):
        idx2 = np.argmin(np.abs(t - (sp + kernel_length/2)))
        x[idx2] = ak[idx]
        
    #filter with kernel 
    x = scipy.signal.convolve(x,kernel,mode = 'same')
    #downsample
    x = x[::over_samp]
    t = t[::over_samp]
    
    return tk, ak, t, x
    
    
    
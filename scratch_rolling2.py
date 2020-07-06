#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:22:00 2020

@author: peter
"""


import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import scipy.signal
import ca_detect_sliding_emom as cad
import generate_e_spline as ges
import double_consistency_search as dc

np.random.seed(10)

fs_fast = 500
fs = 5
length = 10
tau = 1

if fs_fast%fs != 0:
    raise ValueError('Bum')

num_spikes = 0
while num_spikes == 0:
    tk_true,ak_true,t,x = FRIF.make_signal(length,fs_fast,tau = tau)
    num_spikes = len(tk_true)
    
plt.cla()



#now sample with rolling shutter
tophat = np.zeros(2*(fs_fast//fs))
tophat[fs_fast//fs:] = 1/(fs_fast//fs)

x_slow = scipy.signal.convolve(x,tophat,mode = 'same')

winlens = [128,32]
modes = ['estimate','fixed']
fixed_K = 1
spike_thresh = 0
hist_res = 1
hist_thresh = 0.05
box_filter = True
box_filter_length = 1/fs
#x_slow += np.random.normal(scale = 0.1,size = len(x))
all_tk = []
all_ak = []

for idx,win_len in enumerate(winlens):
    if modes[idx] == 'fixed':
        fixed_K_val= fixed_K
    else:
        fixed_K_val = None
            
    if not box_filter:
        tk,ak,_,_ = ca_detect.sliding_window_detect(x, t, win_len, tau, mode = modes[idx], fixed_K = fixed_K_val)
    else:
        tk,ak,_,_ = ca_detect.sliding_window_detect_box_filtered(x, t, win_len, tau, box_filter_length, mode = modes[idx], fixed_K = fixed_K_val)
        
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

jhist = all_hists[0]*all_hists[1]

tk,ak = detect_spikes(jhist, bins, all_tk, all_ak, hist_thresh)
tk,ak,_ = dc.double_consistency_histogram(x_slow, t, tau,hist_thresh = 0.5,winlens = [128,32],spike_thresh = 0,box_filter = False,box_filter_length = 1/fs)

plt.cla()
plt.plot(t,x_slow)
plt.plot(t,x_slow,'.')
plt.stem(tk_true,ak_true,'r', label = 'True',markerfmt='ro')
plt.stem(tk,ak,'k',label = 'Detected',markerfmt ='xk',linefmt = '--k')
plt.legend()
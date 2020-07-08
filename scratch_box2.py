#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:05:40 2020

@author: peter
"""

#A scratch to do decaying exponentials
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import FRI_functions as FRIF


import extract_exponentials as ee

#6np.random.seed(0)

T = 1/10
lam = 0.5
tau = 0.5
length = 40
win_len = 128
oversamp = 64

#Make long signal of decaying
t_k,a_k,t,x = FRIF.make_signal(length,(1/T)*oversamp,firing_rate = lam,tau = tau,spike_std = 0)

#add rolling shutter 
shutter_length = T*6

shutter_fcn = np.zeros(int(np.round(shutter_length*oversamp/T))+2)
shutter_fcn[1:-1] = 1/int(np.round(shutter_length*oversamp/T))

#filter with the shutter
x = scipy.signal.convolve(x,shutter_fcn,mode = 'full')

t = t[::oversamp]
x = x[:-len(shutter_fcn) + 1:oversamp]

noise = np.random.normal(scale = 0.0000*np.max(x),size = len(x))
x += noise

SNR = 10*np.log10(np.sum(x**2)/np.sum(noise**2))
print(f'SNR: {SNR}')

all_tk2,all_ak2 = ee.sliding_window_detect(t,x,tau,win_len,fixed_K = None)

all_tk,all_ak = ee.sliding_window_detect_box_filtered(t,x,tau,win_len,shutter_length,fixed_K = None)

plt.figure()
plt.plot(t,x)




    
#plot 
fig,ax = plt.subplots()


for idx in range(len(all_tk)):
    ax.plot(all_tk2[idx],np.ones(len(all_tk2[idx]))+idx,'xr',alpha = 1)
    
for idx in range(len(all_tk)):
    ax.plot(all_tk[idx],np.ones(len(all_tk[idx]))+idx,'.k')
    

for ti in t_k:
    ax.plot([ti,ti],[0,idx],'r')
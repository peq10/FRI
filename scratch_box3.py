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
import double_consistency_search as dc

#6np.random.seed(0)

T = 1/2
lam = 0.5
tau = 0.5
length = 40
oversamp = 64

#Make long signal of decaying
tk_true,ak_true,t,x = FRIF.make_signal(length,(1/T)*oversamp,firing_rate = lam,tau = tau,spike_std = 0)

#add rolling shutter 
shutter_length = T*2

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



plt.figure()
plt.plot(t,x)

tk1,ak1,_ = dc.double_consistency_histogram(x,t,tau,winlens = [32,8],shutter_length = None)
tk2,ak2,_ = dc.double_consistency_histogram(x,t,tau,winlens = [32,8],shutter_length = shutter_length)

plt.plot(tk_true,ak_true*0+8,'.g')
plt.plot(tk1,ak1*0+7,'.r')
plt.plot(tk2,ak2*0+6,'.k')

if len(tk1) > 0:
    cos_score1 = dc.compare_spike_trains(tk1,tk_true,1,1/T,tau)
    print(cos_score1)

if len(tk2) > 0:
    cos_score2 = dc.compare_spike_trains(tk2,tk_true,1,1/T,tau)
    print(cos_score2)



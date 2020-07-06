#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:07:05 2020

@author: peter
"""

#A scratch file for box filtered devtection.

import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import scipy.signal

fs_fast = 500
fs = 20
length = 10
tau = 1

if fs_fast%fs != 0:
    raise ValueError('Bum')

num_spikes = 0
while num_spikes == 0:
    tk_true,ak_true,t,x = FRIF.make_signal(length,fs_fast,tau = tau)
    num_spikes = len(tk_true)
    
plt.plot(x/x.max())

#now sample with rolling shutter
tophat = np.zeros(2*(fs_fast//fs))
tophat[fs_fast//fs:] = 1

x_slow = scipy.signal.convolve(x,tophat,mode = 'same')

plt.plot(x_slow/x_slow.max())
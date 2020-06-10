#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:56:07 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import scipy.signal
import scipy.linalg
import scipy.special
import scipy.io

#np.random.seed(0)

fs = 6.4
length = 100
tau = 0.5
win_len = 32

num_spikes = 0
while num_spikes == 0:
    tk_true,ak_true,t,x = FRIF.make_signal(length,fs,tau = tau)
    num_spikes = len(tk_true)


all_tk,all_ak,win_idx,K_i = FRIF.sliding_window_detect(x,t,32,tau)






plt.plot(t,x)
plt.stem(tk_corrtime,ak_corrtime,'r',use_line_collection = True)
plt.stem(tk_true,ak_true,'k',use_line_collection = True)
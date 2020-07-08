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

import generate_e_spline as ges
import matrix_pencil as mp
import extract_exponentials as ee

np.random.seed(0)

T = 1/30
lam = 0.5
tau = 0.5
length = 20
win_len = 64

#Make long signal of decaying
t_k,a_k,t,x = FRIF.make_signal(length,1/T,firing_rate = lam,tau = tau,spike_std = 0)

noise = np.random.normal(scale = 0.0001*np.max(x),size = len(x))
x += noise

SNR = 10*np.log10(np.sum(x**2)/np.sum(noise**2))
print(f'SNR: {SNR}')

all_tk,all_ak = ee.sliding_window_detect(t,x,tau,win_len,fixed_K = None)



plt.figure()
plt.plot(t,x)




    
#plot 
fig,ax = plt.subplots()

for idx in range(len(all_tk)):
    ax.plot(all_tk[idx],np.ones(len(all_tk[idx]))+idx,'.k')

for ti in t_k:
    ax.plot([ti,ti],[0,idx],'r')
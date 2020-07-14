#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:47:41 2020

@author: peter
"""

#sliding window dirac detect
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import FRI_functions as FRIF

import generate_e_spline as ges
import get_c_mn_exp as gcm
import matrix_pencil as mp
import extract_exponentials as ee

np.random.seed(0)

oversamp = 64
length = 200
lam = 0.075
T = 1/10
T_s = T/oversamp

noise_level = 0
tau = 3

#Make long signal of approx diracs
t_k,a_k,t,x = FRIF.make_signal(length,1/T_s,firing_rate = lam,tau = tau,spike_std = 0)

x = x[::oversamp]
t = t[::oversamp]

#plt.figure()
#plt.plot(t,x)
win_len = 50


phi,t_phi,c_m_n,n_vec,alpha_vec = ges.decaying_exp_filters(win_len, T, tau)
z_n,t_n = ee.convert_exponential_to_dirac(t,x,phi,t_phi,tau)
all_tk,all_ak = ee.window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec,fixed_K=None)
    
#plot 
#fig,ax = plt.subplots()
plt.cla()

for idx in range(len(all_tk)):
    plt.plot(all_tk[idx],np.ones(len(all_tk[idx]))+idx,'.k')

for ti in t_k:
    plt.plot([ti,ti],[0,idx],'r')
    #plt.plot([ti+T/2,ti+T],[0,idx],'-r',alpha = 0.5)
    
for i in range(0,int(len(t)),2):
    plt.axvspan(t[i],t[(i+1)],facecolor='k', alpha=0.2)
    plt.plot([t[i]+T/2,t[i]+T/2],[0,idx],'k',alpha = 0.1)
    plt.plot([t[i]-T/2,t[i]-T/2],[0,idx],'k',alpha = 0.1)
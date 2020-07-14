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

#np.random.seed(23)

T = 1/10
lam = 0.2
tau = 0.5
length = 10
#over

#Make long signal of decaying
t_k = []
while len(t_k) == 0:
    t_k,a_k,t,x = FRIF.make_signal(length,1/T,firing_rate = lam,tau = tau,spike_std = 0)

#add rolling shutter 
shutter_length = T/10

#shutter_fcn = np.ones(shutter_length/)


win_len = 64

t = t[:win_len]
x = x[:win_len]


plt.figure()
plt.plot(t,x)
plt.plot(t,x,'.')


phi,t_phi,c_m_n,n_vec,alpha_vec = ges.decaying_exp_filters(win_len, T, tau)
z_n,t_n = ee.convert_exponential_to_dirac(t,x,phi,t_phi,tau)


idx_0 = np.argmin(np.abs(t_n))
z_n = z_n[idx_0:idx_0+win_len]
t_n = t_n[idx_0:idx_0+win_len]

#plt.figure()
plt.plot(t_n,z_n/z_n.max())
plt.stem(t_k,a_k,use_line_collection = 'True')



#get s_m
s_m = np.sum(c_m_n*z_n[None,:],-1)

#retrieve t_k,ak
tk,ak = mp.retrieve_tk_ak(s_m, T, alpha_vec,K = None)

tk += n_vec[-1]*T 

plt.stem(tk,ak,'r',use_line_collection = 'True')

print(np.sort(tk) - np.sort(t_k))



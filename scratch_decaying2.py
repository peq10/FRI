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
import get_c_mn_exp as gcm
import matrix_pencil as mp

#np.random.seed(23)

T = 1/10
lam = 0.2
tau = 0.5
length = 10


#Make long signal of decaying
t_k = []
while len(t_k) == 0:
    t_k,a_k,t,x = FRIF.make_signal(length,1/T,firing_rate = lam,tau = tau,spike_std = 0)

win_len = 64

t = t[:win_len]
x = x[:win_len]


plt.figure()
plt.plot(t,x)
plt.plot(t,x,'.')


phi,t_phi,c_m_n,n_vec,alpha_vec = ges.decaying_exp_filters(win_len, T, tau)


def convert_exponential_to_dirac(t,x,phi,t_phi,tau):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    t_phi : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.

    Returns
    -------
    z_n : TYPE
        DESCRIPTION.
    t_n : TYPE
        DESCRIPTION.

    '''
        
    T = np.mean(np.diff(t))
    
    #sample signal with exp. reproducing kernel
    y_n = T*scipy.signal.convolve(x,phi)
    t_y = np.linspace(t[0]+t_phi[0],t[-1]+t_phi[-1],len(y_n))
    
    #reduce to dirac sampling
    z_n = y_n[1:] - y_n[:-1]*np.exp(-T/tau)
    t_n = t_y[1:]

    return z_n,t_n





z_n,t_n = convert_exponential_to_dirac(t,x,phi,t_phi,tau)




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



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

np.random.seed(0)

oversamp = 64
length = 200
lam = 0.1
T = 1
T_s = T/oversamp

noise_level = 0


#Make long signal of approx diracs
t_k,a_k,t,x = FRIF.make_signal(length,1/T_s,firing_rate = lam,tau = 0.001,spike_std = 0)

plt.figure()
plt.plot(t,x)


#generate sampling e-spline
win_len = 16
P = int(win_len/2)
m = np.arange(P + 1)

alpha_0 = -1*np.pi/2
lbda = np.pi/P
alpha_m = alpha_0 +lbda*m
alpha_vec = 1j*alpha_m

phi,t_phi = ges.generate_e_spline(alpha_vec, T_s,T = T, mode = 'anticausal')
h = np.real(phi[::-1])
t_h = -t_phi[::-1]


y = scipy.signal.convolve(x,h)
y = y + np.random.normal(scale = noise_level*np.max(y),size = len(y))
t_y = np.linspace(t[0] + t_h[0],t[-1]+t_h[-1],len(y)) - (t_h[-1] - t_h[0])/2
plt.plot(t_y,y)
plt.plot

#get y_n samples
n_vec = np.arange(win_len)
t_n = n_vec * T

idx = np.nonzero(np.in1d(t_y,t_n))[0]
y_n = y[idx]

plt.plot(t_n,y_n,'.')

sliding_idx = idx[None,:] + np.arange(0,len(y) - idx[-1],oversamp)[:,None]
y_n_sliding = y[sliding_idx]
t_n_sliding = t_y[sliding_idx]

c_m_n = gcm.get_c_mn_exp2(alpha_vec, n_vec, phi, t_phi, T = T)

s_m_sliding = np.sum(c_m_n[:,None,:]*y_n_sliding[None,:,:],-1)
s_m_sliding = np.moveaxis(s_m_sliding,-1,0)

all_tk = []
all_ak = []
for win_idx,win_sm in enumerate(s_m_sliding):
    tk,ak = mp.retrieve_tk_ak(win_sm, T, alpha_vec,K = None,thresh = 0.3)
    all_tk.append(tk + t_n_sliding[win_idx,0] + (t_h[-1] - t_h[0])/2)
    all_ak.append(ak)
    
#plot 
fig,ax = plt.subplots()

for idx in range(len(all_tk)):
    ax.plot(all_tk[idx],np.ones(len(all_tk[idx]))+idx,'.k')

for ti in t_k:
    ax.plot([ti,ti],[0,idx],'r')
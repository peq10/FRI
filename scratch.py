#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:40:17 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
#scratch making exponential kernels

win_len = 32
P = win_len/2
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda * m
fs = 8 
over_samp = 512
T_s = 1/fs/over_samp

N = len(alpha_vec) - 1

t_phi = np.arange(0,1+T_s,T_s)

sub_phi = np.exp(t_phi[:,None] * alpha_vec[None,:])


phi = sub_phi[:,0]
for i in range(sub_phi.shape[1]-1):
    phi = T_s*scipy.signal.convolve(phi,sub_phi[:,i+1])
    
#calculate the time of the vector
num_samples = len(t_phi)*sub_phi.shape[1] - (sub_phi.shape[1] - 1)
t = np.arange(0,num_samples)*T_s
    
plt.plot(phi.real)
plt.show()
plt.plot(phi.imag)
plt.show()
plt.plot(np.abs(phi) - phi.real)
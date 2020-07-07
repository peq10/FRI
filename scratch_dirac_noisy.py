int#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:09:31 2020

@author: peter
"""

#reproduce diracs recovery noiseless
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import generate_e_spline as ges
import get_c_mn_exp as gcm
import f.general_functions as gf
import matrix_pencil as mp

T= 1
t_int = T*8
K = 2
P = 4*K - 1
N = t_int/T + P

T_s = T/64

#get diracs
t_sig = np.arange(0,t_int,T_s)
itk = [180,390]
t_k = t_sig[itk]
a_k = [1.5,3]


#genertae continuous sig
x = np.zeros_like(t_sig)
x[itk] = a_k


#generate e-spline

m = np.arange(P+1)
omega_0= -1*np.pi/2
lbda = np.pi/P
omega_m = omega_0 +lbda*m
alpha_vec = 1j*omega_m

phi,t_phi = ges.generate_e_spline(alpha_vec, T_s,T = T, mode = 'anticausal')
h = np.real(phi[::-1])
t_h = -t_phi[::-1]

n_vec = np.arange(1,N+1)
t_n = n_vec * T
c_m_n = gcm.get_c_mn_exp2(alpha_vec, n_vec, phi, t_phi, T = T)


#compute cont. time sig
y = scipy.signal.convolve(x,h)
y = y + np.random.normal(scale = 0.00001*np.max(y),size = len(y))



t_0 = t_sig[0] + t_h[0]
t_f = t_sig[-1] + t_h[-1]
t_y = np.arange(t_0,t_f+T_s,T_s)

#get y_n samples
idx = np.nonzero(np.in1d(t_y,t_n))[0]
y_n = y[idx]

plt.figure()
plt.plot(t_y,y)
plt.plot(t_n,y_n,'.')

s_m = np.sum(c_m_n*y_n[None,:],-1)

#use matrix pencil to find - noisy case now
tt_k,aa_k = mp.retrieve_tk_ak(s_m,T,alpha_vec)

plt.figure()
plt.stem(tt_k+t_h[-1]/2,aa_k,'k')
plt.stem(t_k+ t_h[-1]/2,a_k,'r')
plt.plot(t_y,y)
plt.plot(t_n,y_n,'.',markersize = 15)
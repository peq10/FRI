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
t_k,a_k,t,x = FRIF.make_signal(length,1/T,firing_rate = lam,tau = tau,spike_std = 0)

win_len = 64

t = t[:win_len]
x = x[:win_len]


plt.figure()
plt.plot(t,x)
plt.plot(t,x,'.')



#generate sampling e-spline
oversamp = 64

P = int(win_len/2)
m = np.arange(P+1)

alpha_0 = -1*np.pi/2
lbda = np.pi/P
alpha_m = alpha_0 +lbda*m
alpha_vec = 1j*alpha_m

phi,t_phi = ges.generate_e_spline(alpha_vec, T/oversamp,T = T,mode = 'symmetric')

#generate psi
alpha = 1/tau
beta_alpha_t, t_beta = ges.generate_e_spline(np.array([-alpha*T]), T/oversamp,T = T,mode = 'causal')
beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))

psi = (T/oversamp)*scipy.signal.convolve(phi,beta_alpha_t)
t_psi = np.linspace(t_phi[0] + t_beta[0],t_phi[-1]+t_beta[-1],len(psi))
                    
#remove oversampling
phi = phi[::oversamp]
t_phi = t_phi[::oversamp]

phi = phi.real

#sample signal
y_n = T*scipy.signal.convolve(x,phi)

t_y = np.linspace(t[0]+t_phi[0],t[-1]+t_phi[-1],len(y_n))


plt.plot(t_y,y_n/y_n.max())

#reduce to dirac sampling
z_n = y_n[1:] - y_n[:-1]*np.exp(-alpha*T)
t_n = t_y[1:]



idx_0 = np.argmin(np.abs(t_n))
z_n = z_n[idx_0:idx_0+win_len]
t_n = t_n[idx_0:idx_0+win_len]

#plt.figure()
plt.plot(t_n,z_n/z_n.max())
plt.stem(t_k,a_k,use_line_collection = 'True')

#get c_m_n
n_vec = np.arange(int(-1*win_len/2),(win_len/2))
c_m_n = gcm.get_c_mn_exp2(alpha_vec, n_vec, psi, t_psi, T = T)

#get s_m
s_m = np.sum(c_m_n*z_n[None,:],-1)

#retrieve t_k,ak
tk,ak = mp.retrieve_tk_ak(s_m, T, alpha_vec,K = None)

tk += n_vec[-1]*T 

plt.stem(tk,ak,'r',use_line_collection = 'True')

print(np.sort(tk) - np.sort(t_k))



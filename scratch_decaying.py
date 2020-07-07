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

np.random.seed(0)

T = 1/10
lam = 0.5
tau = 0.5
length = 20


#Make long signal of decaying
t_k,a_k,t,x = FRIF.make_signal(length,1/T,firing_rate = lam,tau = tau,spike_std = 0)


plt.figure()
plt.plot(t,x)



#generate sampling e-spline
oversamp = 32
win_len = 64
P = int(win_len/2)
m = np.arange(P + 1)

alpha_0 = -1*np.pi/2
lbda = np.pi/P
alpha_m = alpha_0 +lbda*m
alpha_vec = 1j*alpha_m

phi,t_phi = ges.generate_e_spline(alpha_vec, T/oversamp,T = T)

#generate psi
alpha = 1/tau
beta_alpha_t, t_beta = ges.generate_e_spline(np.array([-alpha*T]), T/oversamp,T = T)
psi = (T/oversamp)*scipy.signal.convolve(phi,beta_alpha_t)
t_psi = np.linspace(t_phi[0] + t_beta[0],t_phi[-1]+t_beta[-1],len(psi))
                    
#remove oversampling
phi = phi[::oversamp]
t_phi = t_phi[::oversamp]

#sample signal
y_n = scipy.signal.convolve(x,phi)
t_y = np.linspace(t[0]+t_phi[0],t[-1]+t_phi[-1],len(y_n))

#reduce to dirac sampling
z_n = y_n[1:] - y_n[:-1]*np.exp(-alpha*T)
t_n = t_y[1:] - (t_phi[-1] - t_phi[0])/2

plt.figure()
plt.plot(t_n,z_n)
plt.stem(t_k,a_k)

#get c_m_n
n_vec = np.arange(win_len)
c_m_n = gcm.get_c_mn_exp2(alpha_vec, n_vec, psi, t_psi, T = T)

#get sliding window indices
sliding_idx = n_vec[None,:] + np.arange(len(z_n) - len(n_vec) + 1)[:,None]
z_n_window = z_n[sliding_idx]
t_n_window = t_n[sliding_idx] + n_vec[-1]*T

s_m_window = np.sum(c_m_n[:,None,:]*z_n_window[None,:,:],-1)
s_m_window = np.moveaxis(s_m_window,-1,0)

all_tk = []
all_ak = []
tst = []
for win_idx,win_sm in enumerate(s_m_window):
    tk,ak = mp.retrieve_tk_ak(win_sm, T, alpha_vec,K = None,thresh = 0.3)
    all_tk.append(tk + t_n_window[win_idx,0] - (t_psi[-1] - t_psi[0])/2 )
    all_ak.append(ak)
    tst.append(tk)
    
#plot 
fig,ax = plt.subplots()

for idx in range(len(all_tk)):
    ax.plot(all_tk[idx],np.ones(len(all_tk[idx]))+idx,'.k')

for ti in t_k:
    ax.plot([ti,ti],[0,idx],'r')
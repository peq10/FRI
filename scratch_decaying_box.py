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
import get_c_mn_exp as gcm

#np.random.seed(23)

T = 1/10
lam = 0.2
tau = 0.5
length = 10
oversamp = 32

#Make long signal of decaying
t_k = []
while len(t_k) == 0:
    t_k,a_k,t,x = FRIF.make_signal(length,(1/T)*oversamp,firing_rate = lam,tau = tau,spike_std = 0)

#add rolling shutter 
shutter_length = T*10

shutter_fcn = np.zeros(int(np.round(shutter_length*oversamp/T))*2)
shutter_fcn[:int(np.round(shutter_length*oversamp/T))] = 1/int(np.round(shutter_length*oversamp/T))

#filter with the shutter
x = scipy.signal.convolve(x,shutter_fcn,mode = 'full')


t = t[::oversamp]
x = x[::oversamp]


win_len = 64

t = t[:win_len]
x = x[:win_len]


plt.figure()
plt.plot(t,x)
plt.plot(t,x,'.')

oversamp = 64

alpha_vec = ges.make_alpha_vec(win_len)
phi,t_phi = ges.generate_e_spline(alpha_vec, T/oversamp,T = T,mode = 'symmetric')

#generate psi
alpha = 1/tau
beta_alpha_t, t_beta = ges.generate_e_spline(np.array([-alpha*T]), T/oversamp,T = T,mode = 'causal')
beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))
psi = (T/oversamp)*scipy.signal.convolve(phi,beta_alpha_t)

#generate shutter fcn
shutter_fcn = np.zeros(int(np.round(shutter_length*oversamp/T))*2)
shutter_fcn[:int(np.round(shutter_length*oversamp/T))] = 1/int(np.round(shutter_length*oversamp/T))
shutter_t = np.linspace(0,shutter_length*2,len(shutter_fcn))

#add shutter to psi
psi = scipy.signal.convolve(psi,shutter_fcn)

t_psi = np.linspace(t_phi[0] + t_beta[0] + shutter_t[0],t_phi[-1]+t_beta[-1] +shutter_t[-1],len(psi))
                    
#remove oversampling
phi = phi[::oversamp]
t_phi = t_phi[::oversamp]
phi = phi.real

#get c_m_n
n_vec = np.arange(int(-1*win_len/2),int(win_len/2))
c_m_n = gcm.get_c_mn_exp2(alpha_vec, n_vec, psi, t_psi, T = T)
    
    
z_n,t_n = ee.convert_exponential_to_dirac(t,x,phi,t_phi,tau)






idx_0 = np.argmin(np.abs(t_n))
z_n = z_n[idx_0:idx_0+win_len]
t_n = t_n[idx_0:idx_0+win_len]

#plt.figure()
plt.plot(t_n,z_n/z_n.max())
plt.stem(t_k,a_k,use_line_collection = 'True')



#get s_m
s_m = np.sum(c_m_n*z_n[None,:],-1)
uk = mp.matrix_pencil(s_m,K = 2)
#retrieve t_k,ak
tk,ak = mp.retrieve_tk_ak(s_m, T, alpha_vec,K = None,remove_negative=True)

tk += n_vec[-1]*T -1*shutter_length

plt.stem(tk,ak,'r',use_line_collection = 'True')

print(np.sort(tk) - np.sort(t_k))



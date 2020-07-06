#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:40:17 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.special
import generate_e_spline
import scipy.io

data = scipy.io.loadmat('/home/peter/python/ca_transient/ca_transient/data/real_data.mat')



sp = np.squeeze(data['sp'])
original_signal = np.squeeze(data['original_signal'])
original_t = np.squeeze(data['original_t'])
T_s = float(data['T_s'])
tau = float(data['tau'])

f_s = 1 / T_s
T   = T_s
len_ = original_t[-1] - original_t[0]
win_len = 32

win_len = 32;

T_s     = np.mean(np.diff(original_t))
TTs     = T / T_s
N       = win_len
tot_len = len(original_t)


#scratch making c_m_n exponential 

over_samp = 512
T_s2      = T / over_samp

P            = N / 2
m            = np.arange(0,P+1)
alpha_0      = -1j * np.pi / 2
lbda       = 1j * np.pi / P
alpha_vec    = alpha_0 + lbda * m
phi, t_phi = generate_e_spline.generate_e_spline(alpha_vec, T_s2, T);
t_diric = np.arange(0,P+1 + T_s2/T,T_s2/T)*2*np.pi/(P+1)
t_diric = t_diric - (t_diric[-1]-t_diric[0])/2
b       = scipy.special.diric(t_diric, (P+1))
phi     = b.real
phi_test = np.squeeze(scipy.io.loadmat('./phi_enom.mat')['phi'])
np.testing.assert_allclose(phi,phi_test)


# Compute psi(t) = beta_-alphaT(t) * phi(t)
alpha                 = 1 / tau
beta_alphaT, t_beta = generate_e_spline.generate_e_spline(np.array([-alpha*T]), T_s2, T = T)
beta_alphaT_rev       = np.concatenate(([0],beta_alphaT[:0:-1]))
t_beta_rev            = -t_beta[::-1]
t_0                   = t_phi[0] + t_beta_rev[0]
t_end                 = t_phi[-1] + t_beta_rev[-1]
psi                   = T_s2 * scipy.signal.convolve(phi, beta_alphaT_rev)
t_psi                 = np.arange(t_0,t_end +T_s2,T_s2)
psi_test = np.squeeze(scipy.io.loadmat('./psi_enom.mat')['psi'])
np.testing.assert_allclose(psi,psi_test)


if N%2 == 0:
    n1 = int(-N/2)
    n2 = int(N/2) - 1
else:
    n1 = -(N-1)/2
    n2 = (N-1)/2

n_vec = np.arange(n1,n2+1)
t1    = n1 * T
t2    = (n2+1) * T - T_s
t     = np.arange(t1,t2+T_s,T_s)

cmn_input_tuple = (alpha_vec, n_vec, psi, t_psi, T)
np.save('./cmn_input.npy',cmn_input_tuple)

#c_m_n = get_c_m_n_exp(*cmn_input_tuple)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:50:59 2020

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import f.general_functions as gf
import scipy.special
import get_c_mn_exp as gcm
#code to check my CMN is correct

fs = 2
over_samp = 8

N = 32

P = 14
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda*m

phi, t_phi = FRIF.generate_e_spline(alpha_vec, 1/fs/over_samp)

#t_diric = np.arange(-(P+1)/2,(P+1)/2 +1/over_samp,1/fs/over_samp)*(2*np.pi)/(P+1)
#phi = scipy.special.diric(t_diric, P+1)
#t_phi = t_diric
#phi = phi.real


#get the reproduciton from the kernel
i = 0

num = len(phi)

#try and recover the first of alpha vec


n_vec = (np.arange(0,30)).astype(int)
t_r = np.arange(n_vec[0]/fs + t_phi[0],n_vec[-1]/fs + t_phi[-1],1/fs/over_samp)
sig = np.zeros_like(t_r).astype(np.complex128)
sig2 = np.zeros_like(sig)

c_m_n = gcm.get_c_mn_exp(alpha_vec, n_vec, phi, t_phi,debug = True)
c_m_n_2 = gcm.get_c_mn_exp2(alpha_vec, n_vec, phi, t_phi)

corr_exp = np.exp(alpha_vec[i]*t_r)
plt.cla()
plt.plot(t_r,corr_exp.real)


for idx,n in enumerate(n_vec):
    idx2 = int(idx*fs*over_samp)
    try:
        sig[idx2:idx2+int(len(phi))] += c_m_n[i,idx]*phi
        plt.plot(t_r[idx2:idx2+int(len(phi))],(c_m_n_2[i,idx]*phi).real,'k',alpha = 0.1)
        sig2[idx2:idx2+int(len(phi))] += c_m_n_2[i,idx]*phi
    except ValueError:
        break
    
    #plt.plot(t_r[idx0+idx2:idx0+idx2+len(phi)],(c_m_n[0,idx]*phi).real,'k',alpha = 0.5)

plt.plot(t_r,sig2.real)
plt.plot(t_r[::over_samp],sig2.real[::over_samp],'.')



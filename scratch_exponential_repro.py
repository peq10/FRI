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
#code to check my CMN is correct

fs = 10
over_samp = 512

N = 32

P = 14
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda*m

phi, t_phi = FRIF.generate_e_spline(alpha_vec, 1/fs/over_samp)




#get the reproduciton from the kernel
i = 0

num = len(phi)
sig = np.zeros(num*5).astype(np.complex128)
t_r = (np.arange(len(sig))- 3*len(phi))/fs/over_samp
#try and recover the first of alpha vec
corr_exp = np.exp(alpha_vec[i]*t_r)
plt.cla()
plt.plot(t_r,corr_exp.real)

n_vec = (np.arange(-45,15)).astype(int)
c_m_n = FRIF.get_c_mn_exp(alpha_vec, n_vec, phi, t_phi)
c_m_n_2 = FRIF.get_c_mn_exp2(alpha_vec, n_vec, phi, t_phi)

for idx,n in enumerate(n_vec):
    idx0 = 3*len(phi)
    idx2 = int((n)*fs*over_samp)
    sig[idx0+idx2:idx0+idx2+int(len(phi))] += c_m_n_2[i,idx]*phi
    
    #plt.plot(t_r[idx0+idx2:idx0+idx2+len(phi)],(c_m_n[0,idx]*phi).real,'k',alpha = 0.5)

plt.plot(t_r,sig.real)
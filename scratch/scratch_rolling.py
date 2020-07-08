#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:07:05 2020

@author: peter
"""

#A scratch file for box filtered devtection.

import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import scipy.signal
import ca_detect_sliding_emom as cad
import generate_e_spline as ges
import get_c_mn_exp as gce
import extract_decaying_exponentials as ede
import scipy.linalg

np.random.seed(0)

fs_fast = 500
fs = 10
length = 6
tau = 1

if fs_fast%fs != 0:
    raise ValueError('Bum')

num_spikes = 0
while num_spikes == 0:
    tk_true,ak_true,t,x = FRIF.make_signal(length,fs_fast,tau = tau)
    num_spikes = len(tk_true)
    
plt.cla()


#now sample with rolling shutter
tophat = np.zeros(2*(fs_fast//fs))
tophat[fs_fast//fs:] = 1/(fs_fast//fs)

x = scipy.signal.convolve(x,tophat,mode = 'same')


T = np.mean(np.diff(t))

#construct sampling kernel
over_samp = 512
win_len = 1000
mode = 'estimate'
N = win_len
P = N/2
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda*m

#for some reason we don't actually use phi, just the time stamps??
phi, t_phi = ges.generate_e_spline(alpha_vec, T/over_samp, T = T)
#t_diric = np.arange(-(P+1)/2,(P+1)/2 +1/over_samp,1/over_samp)*(2*np.pi)/(P+1)
#phi = scipy.special.diric(t_diric, P+1)
#phi = phi.real
    
#generate psi
alpha = 1/tau
beta_alpha_t, t_beta = ges.generate_e_spline(np.array([-alpha*T]), 1/over_samp)
beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))
psi = (T/over_samp)*scipy.signal.convolve(phi,beta_alpha_t)


shutter_length = 1/fs

#add in the effect of the rolling shutter filter
shutter_fcn = np.zeros(int(np.round(shutter_length/np.mean(np.diff(t_phi))))+2)
shutter_fcn[1:-1] = 1
shutter_fcn = shutter_fcn / np.sum(shutter_fcn)
#convolve psi with the shutter fcn
psi = scipy.signal.convolve(psi,shutter_fcn)
#plt.plot(psi)
t_psi = np.arange(len(psi))*T/512


#now downsample phi, t_phi to remove oversamplign
phi = phi[::over_samp]
t_phi = t_phi[::over_samp]
tst = scipy.signal.convolve(x,phi,mode = 'same')

tst2 = tst[1:] - tst[:-1]*np.exp(-T/tau)

sig = tst2[:1000]

plt.plot(sig)

#calculate signal moments
n_vec = np.arange(len(sig))
c_m_n = gce.get_c_mn_exp2(alpha_vec,n_vec,psi,t_psi,T = T)

sm = np.sum(c_m_n*sig[None,:],-1)


#locate the diracs
K = 2
S = scipy.linalg.toeplitz(sm[K+1:],sm[K+1::-1])
_,_,V = scipy.linalg.svd(S)
h = V[:,-1]
uu_k = np.roots(h)
pp_k = np.mod(np.angle(uu_k),2*np.pi)
tt_k = T*pp_k/lbda

'''
#get cmn coefficients
if N%2 == 0:
    n_vec = np.arange(-int(N/2),int(N/2))
else:
    n_vec = np.arange(-int((N-1)/2),int((N+1)/2))
    

c_m_n = gce.get_c_mn_exp2(alpha_vec,n_vec,psi,t_psi,T = T)


#iterate through the vector x and detect expoentials in sliding window
K_i = np.zeros(len(x) - win_len)
all_tk = []
all_ak = []
win_idx = []

def circular_convolution(a,b):
    return np.convolve(np.concatenate((a,a)),b)[len(b)-1:len(b)-1+len(a)]
    

for i_0 in range(len(x) - win_len):
    x_part = x[i_0:i_0+win_len]
    t_part = t[i_0:i_0+win_len]
    
    if mode == 'estimate':
        tk,ak = ede.extract_decaying_exponentials(x_part, t_part, 
                                                  tau, phi, t_phi,
                                                  alpha_0, lbda, T,
                                                  c_m_n, n_vec, K = None)
        
        #sampling period. this is just T in all cases as far as I can tekk
        t_s = t_part[1] - t_part[0]
        
        #now make x such that it is sampled by exponential repro.
        #I don't really know why its a circular convolution as opposed to normal convolution.
        y_n = t_s*circular_convolution(x_part,phi)
    
        z_n = y_n[1:] - y_n[:-1]*np.exp(-T/tau)
        
        if np.any(ak > 5000):
            break
            #pass
        
    elif mode == 'fixed':
        if fixed_K is None:
            raise ValueError('Must provide fixed_K for mode == fixed')
        tk,ak = ede.extract_decaying_exponentials(x_part, t_part, 
                                                  tau, phi, t_phi,
                                                  alpha_0, lbda, T,
                                                  c_m_n, n_vec, K = fixed_K)

    #remove negative spikes?
    pos_sp = ak >= 0
    tk = tk[pos_sp]
    ak = ak[pos_sp]
    
    all_tk.append(tk)
    all_ak.append(ak)
    win_idx.append(i_0*np.ones(len(tk)))
    K_i[i_0] = len(tk)


all_ak = np.concatenate(all_ak)
all_tk = np.concatenate(all_tk)


plt.plot(t,x)
plt.stem(all_tk,all_ak,'r',use_line_collection = True)
plt.stem(tk_true,ak_true,'k',use_line_collection = True)

'''
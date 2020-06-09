#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:56:07 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import scipy.signal
import scipy.linalg
import scipy.special
import scipy.io

np.random.seed(0)

fs = 6.4
length = 5
tau = 0.5
win_len = 32

num_spikes = 0
while num_spikes == 0:
    tk_true,ak_true,t,x = FRIF.make_signal(length,fs,tau = tau)
    num_spikes = len(tk_true)

plt.plot(t,x)
plt.show()

mdict = {'f':x,'t':t,'sp':tk_true,'tau':tau}

scipy.io.savemat('./peter_data.mat',mdict)

#first filter with an exponential reproducing kernel
#generate kernel
over_samp = 512
N = win_len
P = N/2
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda*m

phi, t_phi = FRIF.generate_e_spline(alpha_vec, 1/fs/over_samp,T = 1/fs)
t_diric = np.arange(-(P+1)/2,(P+1)/2 +1/over_samp,1/over_samp)*(2*np.pi)/(P+1)
phi = scipy.special.diric(t_diric, P+1)
phi = phi.real


plt.plot(phi)
plt.show()

#generate ps
alpha = 1/tau
beta_alpha_t, t_beta = FRIF.generate_e_spline(np.array([-alpha*(1/fs)]), 1/over_samp)
beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))
psi = (1/fs/over_samp)*scipy.signal.convolve(phi,beta_alpha_t)
t_psi = np.arange(len(psi))/fs/512

plt.plot(psi)
plt.show()


#get cmn coefficients
if N%2 == 0:
    n_vec = np.arange(-int(N/2),int(N/2))
else:
    n_vec = np.arange(-int((N-1)/2),int((N+1)/2))
    
c_m_n_input = scipy.io.loadmat('./data/c_m_n_input.mat')
c_m_n = FRIF.get_c_mn_exp2(alpha_vec,n_vec,np.squeeze(c_m_n_input['psi']),np.squeeze(c_m_n_input['t_psi']))

test_dict = scipy.io.loadmat('./data/input_extract_decaying.mat')
c_m_n = np.squeeze(test_dict['c_m_n'])

#now downsample phi, t_phi to remove oversamplign
phi = phi[::over_samp]
t_phi = t_phi[::over_samp]


#save all input to extract_decaying_exponential
dec_dict = {'x':x,'t_x':t,'alpha_vec':alpha_vec,'tau':tau,'phi':phi,'t_phi':t_phi,'alpha_0':alpha_0,'lbda':lbda,'T':1/fs,'c_m_n':c_m_n,'n_vec':n_vec,'K':2}
np.save('./data/dec_input.npy',dec_dict)

#now neeed to compute yn as <x,phi(t/T - n)>
#is not a convolution
y_n = (1/fs)*scipy.signal.convolve(x,phi[::-1], mode = 'same')
#need to make signal as if it were sampled with exponential repro kernel



plt.plot(y_n)
plt.show()

z_n = y_n[1:] - y_n[:-1]*np.exp(-1/(tau*fs))

plt.plot(z_n)
plt.show()

#compute s_m
#test = scipy.io.loadmat('./c_m_n_corr.mat')['c_m_n']
#c_m_n = np.copy(test)

s_m = np.sum(c_m_n[:,1:]*z_n[None,:],-1)

sm_test = scipy.io.loadmat('./data/sm.mat')  
s_m = np.squeeze(sm_test['s_m'])
kk = int(np.floor(len(s_m)/2) + 1)


#estimating K from S
S = scipy.linalg.toeplitz(s_m[kk:],s_m[kk::-1])
_,s,_ = scipy.linalg.svd(S)
K = np.sum(s/s[0] > 0.3)
print(len(tk_true))
print(K)

#calculate u_k using matrix pencil

M = int(np.round(P/2))
P = int(P)
p = 1
s_m = np.squeeze(sm_test['s_m'])


H_test = scipy.io.loadmat('./data/H.mat')
uk_test = np.squeeze(H_test['uk'])


u_k = FRIF.acmp_p(s_m, K, M, P, p)

out_test = scipy.io.loadmat('./data/tk.mat')

tk = np.real(((1/fs) * np.log(u_k) / lbda))

#find amplitudes ak
A = np.zeros((K,K)).astype(np.complex128)
for i in range(K):
    A[i,:] = u_k[:K]**i
    
B = s_m[:K]
b_k = scipy.linalg.solve(A,B)
ak = np.real(b_k*np.exp(-alpha_0*tk*fs))


#shift to correct time
tk -= n_vec[0]/fs


plt.figure()
plt.plot(t,x)
plt.stem(tk,ak,'r')
plt.stem(tk_true,ak_true,'k')
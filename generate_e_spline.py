#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:31:17 2020

@author: peter
"""
import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt


def generate_e_spline(alpha_vec,T_s,T = 1,mode = 'causal'):
    '''
    Re-written in Python from Jon Onativia's Original.
    https://doi.org/10.1088/1741-2560/10/4/046017
    
    Code from http://www.schultzlab.org/software/index.html
    
    Generate the exponential spline of order P+1 corresponding to a vector of
    alpha values and with a given temporal resolution. The resulting spline 
    is obtained in time domain computing the P convolutions of the P+1 zero 
    order E-splines:
    phi_a_vec(t) = phi_a_0(t) * phi_a_1(t) * ... * phi_a_N(t)
    Parameters
    ----------
    alpha_vec : TYPE
    Vector of P+1 alpha values of the E=spline.
    T_s : TYPE
    Time resolution of the spline.
    T : TYPE
    Scale factor. Default T = 1.
    mode : TYPE
    Optional argument. 'causal', 'symmetric' or 'anticausal'. Default 'causal'.
    
    Returns
    -------
    phi       : Vector of size (P+1)/T + 1 with the values of the
    E-spline.
    t         : Time stamps of the corresponding values of the phi vector.
    
    '''
       
    #apply scaling
    T_s /= T
    
    #generate the base splines to convolve from the alpha parameters
    t_phi = np.arange(0,1+T_s,T_s)
    sub_phi = np.exp(t_phi[:,None] * alpha_vec[None,:])
    sub_phi[-1,:] = 0
    
    #convolve them all together
    phi = np.concatenate(([0],sub_phi[:-1,0]))
    for i in range(sub_phi.shape[1]-1):
        phi = T_s*scipy.signal.convolve(phi,sub_phi[:,i+1])
       
    #calculate the time of the vector
    #as each convolution is len(a) + len(b) - 1
    num_samples = len(t_phi)*sub_phi.shape[1] - (sub_phi.shape[1] - 1)
    t = np.arange(0,num_samples)*T_s
    t *= T
    
    # TODO -  add the mode modifications to t
    if mode == 'symmetric':
        idx_max = np.argmax(phi)
        t = t - t[idx_max]
    elif mode == 'anticausal':
        phi = phi[::-1]
        t = -t[::-1]
    elif mode == 'causal':
        pass
    else:
        raise ValueError('Mode not recognised')
    
    return phi, t
   
def test_e_spline():
    correct = np.squeeze(scipy.io.loadmat('./phi.mat')['phi'])
    
    T_s = float(scipy.io.loadmat('./T_s.mat')['T_s'])
    alpha_vec = np.squeeze(scipy.io.loadmat('./alpha_vec.mat')['alpha_vec'])
    ours,_ = generate_e_spline(alpha_vec,T_s)
    
    #plt.plot(np.abs((correct.imag - ours.imag)/ours.imag)[10000:-10000]*100)
    
    plt.plot(correct.real)
    plt.plot(ours.real)
    dec = -1*(np.log10(np.max(np.abs(ours)) - np.min(np.abs(ours))) - 7)
    np.testing.assert_almost_equal(correct,ours, decimal = dec)
    
if __name__ == '__main__':
    test_e_spline()
    
    
    
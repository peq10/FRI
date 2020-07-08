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
import get_c_mn_exp as gcm

def make_alpha_vec(win_len):
    P = int(win_len/2)
    m = np.arange(P+1)
    alpha_0 = -1*np.pi/2
    lbda = np.pi/P
    alpha_m = alpha_0 +lbda*m
    alpha_vec = 1j*alpha_m
    return alpha_vec

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
   
def decaying_exp_filters(win_len,T,tau,oversamp = 64):
    '''
    Generates the sampling filters and exponential reproducing filters for 
    converting decaying exponentials into diracs and then recovering their
    positions and amplitudes.

    Parameters
    ----------
    win_len : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    oversamp : TYPE, optional
        DESCRIPTION. The default is 64.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    phi : TYPE
        DESCRIPTION.
    t_phi : TYPE
        DESCRIPTION.
    c_m_n : TYPE
        DESCRIPTION.
    n_vec : TYPE
        DESCRIPTION.
    alpha_vec : TYPE
        DESCRIPTION.

    '''
    if win_len%2 != 0:
        raise NotImplementedError('Only implemented even length windows')
    
    alpha_vec = make_alpha_vec(win_len)
    phi,t_phi = generate_e_spline(alpha_vec, T/oversamp,T = T,mode = 'symmetric')
    
    #generate psi
    alpha = 1/tau
    beta_alpha_t, t_beta = generate_e_spline(np.array([-alpha*T]), T/oversamp,T = T,mode = 'causal')
    beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))
    psi = (T/oversamp)*scipy.signal.convolve(phi,beta_alpha_t)
    t_psi = np.linspace(t_phi[0] + t_beta[0],t_phi[-1]+t_beta[-1],len(psi))
                        
    #remove oversampling
    phi = phi[::oversamp]
    t_phi = t_phi[::oversamp]
    phi = phi.real
    
    #get c_m_n
    n_vec = np.arange(int(-1*win_len/2),int(win_len/2))
    c_m_n = gcm.get_c_mn_exp2(alpha_vec, n_vec, psi, t_psi, T = T)
    
    return phi,t_phi,c_m_n,n_vec,alpha_vec 

def box_decaying_exp_filters(win_len,T,tau,shutter_length,oversamp = 64):
    '''
    For box filtered

    Parameters
    ----------
    win_len : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    shutter_length : TYPE
        DESCRIPTION.
    oversamp : TYPE, optional
        DESCRIPTION. The default is 64.

    Returns
    -------
    phi : TYPE
        DESCRIPTION.
    t_phi : TYPE
        DESCRIPTION.
    c_m_n : TYPE
        DESCRIPTION.
    n_vec : TYPE
        DESCRIPTION.
    alpha_vec : TYPE
        DESCRIPTION.

    '''
    alpha_vec = make_alpha_vec(win_len)
    phi,t_phi = generate_e_spline(alpha_vec, T/oversamp,T = T,mode = 'symmetric')
    
    #generate psi
    alpha = 1/tau
    beta_alpha_t, t_beta = generate_e_spline(np.array([-alpha*T]), T/oversamp,T = T,mode = 'causal')
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
    
    return phi,t_phi,c_m_n,n_vec,alpha_vec 
   
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
    
    
    
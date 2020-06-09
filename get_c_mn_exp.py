#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:07 2020

@author: peter
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def get_c_mn_exp(alpha_m, n, phi, t_phi, T = 1):
    '''
    USAGE:
      c_m_n = get_c_m_n_exp(alpha_m, n, phi, t[, T, t_0])
    
     INPUT:
      - alpha_m : Vector of size M with the parameters of the exponentials to 
                  be reproduced.
      - n       : Vector of size N with the values where the summation will be
                  evaluated.
      - phi     : Exponential reproducing kernel.
      - t_phi   : Time stamps of the kernel.
      - T       : Optional argument. Scale factor. Default T = 1.
      - t_0     : Optional argument. t value where c_m_0 will be evaluated. Default t_0 = 0.
    
     OUTPUT:
      - c_m_n   : Coefficients to reproduce the exponentials.
    '''
    #calculate cm0
    idx = np.round((len(phi)-1)/(n.max() - n.min())*(n - n.min())).astype(int) 
    c_m_0 = 1/(np.sum(np.exp(alpha_m[:,None] * n[None,:])*phi[idx][None,:],-1))
    
    # Compute the remaining c_m_n from c_m_0
    exp_mat = np.exp(alpha_m[:,None] * n[None,:])
    c_m_n = exp_mat * c_m_0[:,None]
    
    return c_m_n
        
    
def load_check_input():

    alpha_vec, n_vec, psi, t_psi, T = tuple(np.load('./cmn_input.npy',allow_pickle = True))
    obj = scipy.io.loadmat('./corr_cmn_input.mat')
    
    np.testing.assert_equal(alpha_vec,np.squeeze(obj['alpha_vec']))
    np.testing.assert_equal(n_vec,np.squeeze(obj['n_vec']))
    np.testing.assert_allclose(psi,np.squeeze(obj['psi']))
    np.testing.assert_allclose(t_psi,np.squeeze(obj['t_psi']),atol = 10**-10)
    np.testing.assert_equal(T,np.squeeze(obj['T']))
    
    
    return alpha_vec, n_vec, psi, t_psi, T


def test():
    corr_c_m_n = np.squeeze(scipy.io.loadmat('./c_m_n_corr.mat')['c_m_n'])
    input_ = load_check_input()
    c_m_n = get_c_mn_exp(*input_)
    #currently fails as my cmn are shifted?
    np.testing.assert_allclose(corr_c_m_n,c_m_n)


def get_c_mn_exp2(alpha_m, n, phi, t_phi, T = 1):
    #I removed T and t0 to try and understand better
    t_phi_sampling = np.mean(np.diff(t_phi))

    #get kernel boundaries
    t_1 = t_phi[0]/T
    t_2 = t_phi[-1]/T
    
    sta = np.ceil(np.round(-1*t_2,decimals = 3)).astype(int)
    sto = np.floor(np.round(-1*t_1,decimals = 3)).astype(int)
    npoints = sto - sta + 1
    #compute C_m_0 - I don't understand why the index here is not the full P?
    #time span of the t_phi vector without scaling?    
    l = np.linspace(sta,sto,npoints)
    idx = np.round(-1*T*(t_1 +l)/t_phi_sampling).astype(int)

    phi_l = phi[idx]
    num = np.exp(alpha_m * 0)
    den = np.sum(np.exp(alpha_m[:,None] * l[None,:])*phi_l[None,:],-1)
    c_m_0 = num/den
    
    exp_mat = np.exp(alpha_m[:,None]*n[None,:])
    c_m_n = exp_mat*c_m_0[:,None]
    
    return c_m_n


def test2():
    c_m_n_input = scipy.io.loadmat('./data/c_m_n_input.mat')
    
    alpha_vec = np.squeeze(c_m_n_input['alpha_vec'])
    n_vec = np.squeeze(c_m_n_input['n_vec'])
    psi = np.squeeze(c_m_n_input['psi'])
    t_psi = np.squeeze(c_m_n_input['t_psi'])
    T = np.squeeze(c_m_n_input['T'])
    c_m_n = get_c_mn_exp2(alpha_vec,n_vec,psi,t_psi,T = T)
    
    test_dict = scipy.io.loadmat('./data/input_extract_decaying.mat')
    c_m_n_test = np.squeeze(test_dict['c_m_n'])
    
    try:
        np.testing.assert_allclose(c_m_n,c_m_n_test)
    except AssertionError as err:
        print(err)
        plt.imshow(np.abs(c_m_n_test - c_m_n))
        plt.colorbar()
        print('\n Returning input and corr outpur to workspace ')
        return c_m_n_test,alpha_vec,n_vec,psi,t_psi,T

if __name__ == '__main__':
    ret = test2()
    if ret is not None:
        c_m_n_corr = ret[0]
        alpha_m, n, phi, t_phi, T = ret[1:]


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:07 2020

@author: peter
"""
import numpy as np
        
def get_c_mn_exp2(alpha_m, n, phi, t_phi, T = 1):
    '''
    

    Parameters
    ----------
    alpha_m : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    t_phi : TYPE
        DESCRIPTION.
    T : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    c_m_n : TYPE
        DESCRIPTION.

    '''

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


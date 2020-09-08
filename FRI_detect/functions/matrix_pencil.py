#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:02:11 2020

@author: peter
"""
import scipy.linalg
import numpy as np

def matrix_pencil(sm,K = None, thresh = 0.3):
    '''
    Noisy matrix pencil method for retrieving uk from sample moments (see 
    http://www.commsp.ee.ic.ac.uk/~pld/group/PhDThesis_Onativia15.pdf p.50)
    https://doi.org/10.25560/49792 
    
    Always splits the samples in the middle although that may change in future. 
    Can estimate number of diracs.

    Parameters
    ----------
    sm : complex 1d array
        Signal moments.
    K : Int, optional
        Number of diracs in the signal. The default is None.
        Is estimated if None is given.
    thresh : float, optional
        Threshold for estimating K from signal moments. The default is 0.3.

    Returns
    -------
    uu_k : Complex array length K
        Values from which tk and ak can be calculated.

    '''
    
    M = np.ceil(len(sm)/2).astype(int)
    S = scipy.linalg.toeplitz(sm[M:],sm[M::-1])
    U,s,_, = scipy.linalg.svd(S)    
    
    if K is None:
        
        if s.max() == 0:
            return np.array([])
        else:
            s = s/s.max()
            K = np.sum(s>thresh)
    
    U = U[:,:K]

    S0 = U[1:,:]
    S1 = U[:-1,:]
    Z = np.matmul(scipy.linalg.pinv(S1),S0)
    
    uu_k = scipy.linalg.eig(Z)[0]
    
    return uu_k

def matrix_pencil_noiseless(sm,K):
    '''
    Retrieves Uk in noiseless case using the matrix pencil method
    
    See https://doi.org/10.25560/49792 section 2.4
    
    Parameters
    ----------
    sm : 1D array of complex floats
        Signal moments.
    K : Int
        Number of spikes (model order).

    Returns
    -------
    uu_k : complex vector length K
        uk values from which tk and ak can be retrieved.

    '''
    #Form toeplitz matrix and sub matrices
    S = scipy.linalg.toeplitz(sm[K-1:],sm[K-1::-1])
    S0 = S[1:,:]
    S1 = S[:-1,:]
    #solve eigenvalue problem (eq. 2.66 in above reference)
    Z = np.matmul(scipy.linalg.inv(S1),S0)
    uu_k = scipy.linalg.eig(Z)[0]
    return uu_k

def retrieve_tk_ak(sm,T,alpha_vec, K = None,thresh = 0.3, remove_negative = True):
    '''
    Retrieves dirac locations and amplitudes (tk,ak) from an array of signal moments    

    See https://doi.org/10.25560/49792  sec 2.4 & 2.5    

    Parameters
    ----------
    sm : 1D complex float vector
        signal moments calculated from the expon. repro. sampled signal using the
        c_{m,n} coefficients.
    T : Float
        Sampling period.
    alpha_vec : 1D complex array
        vector of sampling kernel exponential reproductions.
    K : int, optional
        Number of diracs in sample. The default is None.
        If None is estimated from the moments.

    Returns
    -------
    t_k : 1D array of real floats
        Estimated dirac times.
    a_k : 1D array of real floats
        Estimated dirac sizes.

    '''
    #calculate uk values using matrix pencil
    uk = matrix_pencil(sm,K = K,thresh = thresh)    
    
    #retrieve tk, uk using eq. 2.64 
    lbda = np.mean(np.real(np.diff(alpha_vec)/1j))
    omega_0 = np.real(alpha_vec[0]/1j)
    K = len(uk)
    
    t_k = np.real(np.log(uk)*T/(1j*lbda))
    A = np.zeros((K,K)).astype(np.complex128)
    for i in range(K):
        A[i,:] = uk**i
    B = sm[:K]
    b_k = scipy.linalg.solve(A,B)
    a_k = np.real(b_k*np.exp(-1j*omega_0*t_k/T))
    
    #Remove negative detected diracs
    if remove_negative:
        pos = a_k > 0
        a_k = a_k[pos]
        t_k = t_k[pos]

    return t_k,a_k



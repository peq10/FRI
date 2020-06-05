#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:02:11 2020

@author: peter
"""
import scipy.linalg
import numpy as np

def acmp_p(sm, K, M, P, p):
    '''
    


    Parameters
    ----------
    sm : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    P : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    uk

    '''
        
    H = scipy.linalg.toeplitz(sm[M+1:P+2],sm[M+1::-1])
    U,_,_ = scipy.linalg.svd(H)
    
    U = U[:,:K]
    
    Z = np.matmul(scipy.linalg.pinv(U[:-p,:]),U[p:,:])
    
    uk = scipy.linalg.eig(Z)[0]**(1/p)
    
    return uk
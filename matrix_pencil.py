#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:02:11 2020

@author: peter
"""
import scipy.linalg
import numpy as np

def acmp_p(sm, K, M, P, p = 1):
    '''
    http://www.commsp.ee.ic.ac.uk/~pld/group/PhDThesis_Onativia15.pdf
    See page 50
    


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
        
    H = scipy.linalg.toeplitz(sm[M:P+1],sm[M::-1])[:,::-1]

    U,_,_ = scipy.linalg.svd(H)
    
    U = U[:,:K]
    
    Z = np.matmul(scipy.linalg.pinv(U[:-p,:]),U[p:,:])
    
    uk = scipy.linalg.eig(Z)[0]**(1/p)
    
    return uk




if __name__ == '__main__':
    pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:38:11 2020

@author: peter
"""

import numpy as np
import ca_detect_sliding_emom as ca_detect

def double_consistency_histogram(x,t,tau,winlens = [32,8],modes = ['estimate','fixed'], fixed_K = 1):
    
    all_tk = []
    all_ak = []

    for idx,win_len in enumerate(winlens):
        if modes[idx] == 'fixed':
            fixed_K = fixed_K
        else:
            fixed_K = None
                
        tk,ak,_,_ = ca_detect.sliding_window_detect(x, t, win_len, tau, mode = modes[idx], fixed_K = fixed_K)
        all_tk.append(tk)
        all_ak.append(ak)

    raise ValueError('Not finished')
    return tk,ak
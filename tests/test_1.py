#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:01:38 2020

@author: peter
"""
import FRI_detect.functions.simulate_spike_train
import FRI_detect.detect_spikes
import numpy as np


def test():
    np.random.seed(0)
    
    tau = 0.5
    tk,_,t,x = FRI_detect.functions.simulate_spike_train.make_signal(20,8,tau = tau,noise_level = 0.1)
    tk_est = FRI_detect.detect_spikes.detect_spikes(t, x, tau)

    corr = np.array([ 1.55345912,  3.29245283,  4.28616352,  7.01886792,  7.88836478,
        8.75786164, 11.61477987, 16.08647799, 17.20440252])
    np.testing.assert_allclose(tk_est,corr)
    
    
    

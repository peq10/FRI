#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:49:10 2020

@author: peter
"""

from setuptools import setup

setup(
    name='FRI',
    version='0.1',
    license='MIT',
    description='A python package for FRI detection of calcium transients.',
    author='Peter Quicke',
    author_email='peter.quicke@gmail.com',
    url='https://github.com/peq10/FRI',
    packages=['FRI_detect',],
    python_requires='!=2.*, >=3.7.*',
    install_requires=[
        'scipy==1.4.*', 'numpy==1.18.*',
    ],
    setup_requires=[
        'pytest-runner',
    ],
)
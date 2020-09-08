# FRI
Python code implementing FRI methods for time series analysis.

This is an extension of Jon Onativia's method detailed in his PhD thesis
(https://doi.org/10.25560/49792) and publication (https://doi.org/10.1088/1741-2560/10/4/046017)
and draws heavily from his MATLAB code (available at http://www.schultzlab.org/software/index.html)

This has been Python-ised and developed to incorporate integrating detectors.

Yet to be implemented:
- Improvements detailed in https://doi.org/10.25560/65702 including
    - Non-instantaneous rise calcium transients.
    - Noise pre whitening
    - Estimation of model order (spike number per window) by prediction error minimisation.


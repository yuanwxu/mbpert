# MBPert: A computational framework for inferring species dynamics and interactions
MBPert is a generic computational framework for inferring species interactions and predicting dynamics in time-evolving ecosystems from perturbation and time-series data. It is developed specifically in the context of microbiota ecology to better understand microbial species interactions and predict future species dynamics, from metagenomics time series and perturbation data. It assumes that underlying species dynamics is approximated by the generalized Lotka Volterra equations and uses machine learning to iteratively optimize the gLV parameters encoding species interactions.

This repository contains all the code to reproduce the results for

1. Simulated data
    - `pp_simu_random_split.py` --- script for splitting perturbation conditions randomly, in the case of "paired before and after measurements under targeted perturbations".
    - `pp_simu_split_by_cond.py` --- script for splitting perturbation conditions by the number of targeted species, in the case of "paired before and after measurements under targeted perturbations".
    - `pp_simu_loso.py` --- script for splitting perturbation conditions based on whether they target species `i` or not (leave-one-species-out), in the case of "paired before and after measurements under targeted perturbations".
    - `sp_simu_multi_starts.py` --- script for single-group time series data, in the case of "temporal dynamics with time-dependent perturbations".
    - `sp_simu_logo.py` --- script for multi-group time series data, in the case of "temporal dynamics with time-dependent perturbations".
    - `post_run_simu.py` --- script for additional plots
      
2.  Benchmarking against the MTIST datasets
    - `run_mtist.py`
    - `post_run_mtist.py`

    The input data is taken from <https://github.com/jsevo/mtist/tree/main/mtist1.0/mtist_datasets>

3.  *C.diff* infected mouse data
    - `prep_mdsine_mbpert.py`
    - `run_mdsine_cdiff.py`
  
    Original data was downloaded from <http://dx.doi.org/10.5281/zenodo.50624>, including `biomass.txt`, `counts.txt`, and `metadata.txt`. They were processed with `prep_mdsine_mbpert.py` to transform the data into a format suitable for MBPert. 

    

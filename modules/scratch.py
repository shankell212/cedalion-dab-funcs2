#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:17:50 2025

@author: smkelley
"""

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.xrutils as xrutils
import cedalion.models.glm as glm
import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd

import json


# import my own functions from a different directory
import sys
import module_plot_DQR as pfDAB_dqr
import module_imu_glm_filter as pfDAB_imu

import pdb



#%%

def preprocess_post_ma_corr(recTmp, p_rec_str, cfg_dataset, cfg_preprocess, filenm, ):
    
    
    
    # Walking filter 
    if cfg_preprocess['cfg_motion_correct']['flag_do_imu_glm']:     # and np.any(recTmp.aux_ts['ACCEL_X'] != 0): # This might not always work?
        recTmp["od_imu"] = pfDAB_imu.filterWalking(recTmp, "od", cfg_preprocess['cfg_motion_correct']['cfg_imu_glm'], filenm, cfg_dataset['root_dir'])
        
        # !!! quantify slopes for walking filter and plot?
        slope_imu = quant_slope(recTmp, "od_imu", True)
        
    # Get the slope of 'od' before motion correction and any bandpass filtering
    slope_base = quant_slope(recTmp, "od", True)
    # FIXME: could have dictionary slope['base'], slope['tddr'], slope['splineSG'] etc
    
    
    # Spline SG
    if cfg_preprocess['cfg_motion_correct']['flag_do_splineSG']:
        recTmp, slope = motionCorrect_SplineSG( recTmp, cfg_preprocess['cfg_bandpass'] )
    else:
        slope = None
    
    
    # TDDR
    if cfg_preprocess['cfg_motion_correct']['flag_do_tddr']:
        recTmp['od_tddr'] = motion_correct.tddr( recTmp['od'] )
    
    if cfg_preprocess['cfg_motion_correct']['flag_do_imu_glm']:
        recTmp['od_imu_tddr'] = motion_correct.tddr( recTmp['od_imu'] )
        recTmp['od_o_imu_tddr'] = motion_correct.tddr( recTmp['od_o_imu'] )
        
#%%        
    
    p_rec_str = 'tddr'
    # Get slopes after TDDR before bandpass filtering
    slope_tddr = quant_slope(recTmp, f'od_{p_rec_str}', False)
    
    
    # GVTD for TDDR before bandpass filtering
    amp_tddr = recTmp[f'od_{p_rec_str}'].copy()
    amp_tddr.values = np.exp(-amp_tddr.values)
    recTmp.aux_ts[f'gvtd_{p_rec_str}'], _ = quality.gvtd(amp_tddr)
    
    
    # bandpass filter od_tddr
    fmin = cfg_preprocess['cfg_bandpass']['fmin']
    fmax = cfg_preprocess['cfg_bandpass']['fmax']
    recTmp[f'od_{p_rec_str}'] = cedalion.sigproc.frequency.freq_filter(recTmp[f'od_{p_rec_str}'], fmin, fmax)
    

    # Convert OD to Conc
    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
        coords={"wavelength": recTmp['amp'].wavelength},
    )
    
    # Convert to conc Conc
    recTmp[f'conc_{p_rec_str}'] = cedalion.nirs.od2conc(recTmp['od_tddr'], recTmp.geo3d, dpf, spectrum="prahl")
    
    # GLM filtering step
    if cfg_preprocess['flag_do_GLM_filter']:
        recTmp = GLM(recTmp, f'conc_{p_rec_str}', cfg_preprocess['cfg_GLM'])
        
        recTmp[f'od_{p_rec_str}_postglm'] = cedalion.nirs.conc2od(recTmp[f'conc_{p_rec_str}'], recTmp.geo3d, dpf)  # Convert GLM filtered data back to OD
        

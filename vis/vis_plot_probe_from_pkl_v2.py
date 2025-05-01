#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:22:44 2025

@author: smkelley
"""

import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion
import xarray as xr
from cedalion import io, nirs, units
import cedalion.vis.plot_probe as vPlotProbe

#%%

#path2results = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/processed_data/"
path2results = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/Shannon/processed_data"

#filname = 	'blockaverage_STS_tddr_GLMfilt_unpruned_CONC.pkl.gz'     # STS
#filname = 'blockaverage_IWHD_imuGLM_tddr_GLMfilt_unpruned_CONC.pkl.gz'   # IWHD

filname = 'blockaverage_BS_tddr_GLMfilt_unpruned_OD.pkl.gz' 

filepath_bl = os.path.join(path2results , filname) 
    
if os.path.exists(filepath_bl):
    with gzip.open(filepath_bl, 'rb') as f:
        groupavg_results = pickle.load(f)
    blockaverage = groupavg_results['blockaverage']
    blockaverage_stderr = groupavg_results['blockaverage_stderr']
    blockaverage_subj = groupavg_results['blockaverage_subj']
    blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    geo2d = groupavg_results['geo2d']
    geo3d = groupavg_results['geo3d']
    print("Blockaverage file loaded successfully!")

else:
    print(f"Error: File '{filepath_bl}' not found!")
    
    
# If blockaverage in OD, convert to conc

if 'OD' in filname:
    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
        coords={"wavelength": blockaverage.wavelength},
    )
    
    # blockavg
    blockaverage = blockaverage.rename({'reltime':'time'})
    blockaverage.time.attrs['units'] = units.s
    blockaverage_conc = nirs.od2conc(blockaverage, geo3d, dpf, spectrum="prahl")
    blockaverage_conc = blockaverage_conc.rename({'time':'reltime'})
    
    # stderr
    blockaverage_stderr = blockaverage_stderr.rename({'reltime':'time'})
    blockaverage_stderr.time.attrs['units'] = units.s
    blockaverage_stderr_conc = nirs.od2conc(blockaverage_stderr, geo3d, dpf, spectrum="prahl")
    blockaverage_stderr_conc = blockaverage_stderr_conc.rename({'time':'reltime'})
    
    tstat_conc = blockaverage_conc / blockaverage_stderr_conc
   
#%% Plot

tstat = blockaverage / blockaverage_stderr   # tstat = blockavg/ noise

vPlotProbe.run_vis(blockaverage = blockaverage_conc, geo2d = geo2d, geo3d = geo3d)






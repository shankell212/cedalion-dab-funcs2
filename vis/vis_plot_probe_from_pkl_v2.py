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

import cedalion.vis.plot_probe as vPlotProbe

#%%

path2results = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/processed_data/"


#filname = 	'blockaverage_STS_tddr_GLMfilt_unpruned_CONC.pkl.gz'     # STS
filname = 'blockaverage_IWHD_imuGLM_tddr_GLMfilt_unpruned_CONC.pkl.gz'   # IWHD
 
filepath_bl = os.path.join(path2results , filname) 
    
if os.path.exists(filepath_bl):
    with gzip.open(filepath_bl, 'rb') as f:
        groupavg_results = pickle.load(f)
    blockaverage_mean = groupavg_results['blockaverage']
    blockaverage_stderr = groupavg_results['blockaverage_stderr']
    blockaverage_subj = groupavg_results['blockaverage_subj']
    blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    geo2d = groupavg_results['geo2d']
    geo3d = groupavg_results['geo3d']
    print("Blockaverage file loaded successfully!")

else:
    print(f"Error: File '{filepath_bl}' not found!")

#%%

tstat = blockaverage_mean/blockaverage_stderr   # tstat = blockavg/ noise

vPlotProbe.run_vis(blockaverage = blockaverage_stderr, geo2d = geo2d, geo3d = geo3d)


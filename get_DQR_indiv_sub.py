#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get DQR for individual subject

Created on Fri Mar 21 10:05:09 2025

@author: smkelley
"""

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils


import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd
from math import ceil

import module_plot_DQR as dqr
import module_load_and_preprocess as preproc

#%%

root_dir = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/" # change to your data directory

subj = '01' 
task = 'IWHD'
runNum = '01'


cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_threshs' : [1, 60]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_threshs' : [1e-5, 0.84], # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : True,
    'flag_use_psp' : False
}


cfg_motion_correct = {
    'flag_do_splineSG' : False, # if True, will do splineSG motion correction
    'splineSG_p' : 0.99, 
    'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,  # !!! This isn't doing anything? - don't think I've added a check in the code so tddr is always done -- added flag in preprocess mod, but will error if false
    'flag_do_imu_glm' : True,
}

cfg_bandpass = { 
    'fmin' : 0.01 * units.Hz, #0.02 * units.Hz,
    'fmax' : 0.5 * units.Hz  #3 * units.Hz
}
        

cfg_preprocess = {
    'flag_prune_channels' : True,  # FALSE = does not prune chans and does weighted averaging, TRUE = prunes channels and no weighted averaging
    'median_filt' : 3, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'flag_do_GLM_filter' : True,
}

#%% Load in data

subDir = os.path.join(root_dir, 'sub-',subj, 'nirs')
run_nm = f'sub-{subj}_task-{task}_run-{runNum}'

snirf_path = os.path.join(subDir, run_nm + '_nirs.snirf' )

# check if the snirf file exists
if not os.path.exists( snirf_path ):
    print( f"Error: File {snirf_path} does not exist" )
else:
    records = cedalion.io.read_snirf( snirf_path ) 
    rec = records[0]

events_path = os.path.join(subDir, run_nm + '_events.tsv' )

# check if the events.tsv file exists
if not os.path.exists( events_path ):
    print( f"Error: File {events_path} does not exist" )
else:
    stim_df = pd.read_csv(events_path, sep='\t' )
    rec.stim = stim_df

#%% Preprocess 

# Preprocess data with median filt
rec = preproc.preprocess( rec, cfg_preprocess['median_filt'] )

# Prune channels
rec, chs_pruned, sci, psp = preproc.pruneChannels( rec, cfg_preprocess['cfg_prune'] )

# Calculate OD 
# if flag pruned channels is True, then do rest of preprocessing on pruned amp, if not then do preprocessing on unpruned data
if cfg_preprocess['flag_prune_channels']:
    rec["od"] = cedalion.nirs.int2od(rec['amp_pruned'])
    rec.aux_ts["gvtd"], _ = quality.gvtd(rec['amp_pruned'])
    
else:
    rec["od"] = cedalion.nirs.int2od(rec['amp'])
    rec.aux_ts["gvtd"], _ = quality.gvtd(rec['amp'])
    


    
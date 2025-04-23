#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:25:30 2025

@author: smkelley
"""

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils
from cedalion.sigdecomp.ERBM import ERBM

import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd
from math import ceil

import gzip
import pickle
import json


#%%
cfg_hrf = {
    'stim_lst' : ['STS'], 
    't_pre' : 5 *units.s, 
    't_post' : 33 *units.s
    #'t_post' : [ 33, 33 ] *units.s   # !!! GLM does not let you have different time ranges for diff stims right now
    }

cfg_dataset = {
    'root_dir' : "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/",
    'subj_ids' : ['01','02','03','04','05','06','07','08','09','10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],
    'file_ids' : ['STS_run-01'],
    'subj_id_exclude' : ['10', '15', '16', '17'], #['05','07'] # if you want to exclude a subject from the group average
    'cfg_hrf' : cfg_hrf
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]



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

cfg_imu_glm = {'statesPerDataFrame' : 89,   # FOR WALKING DATA
		'hWin' : np.arange(-3,5,1), # window for impulse response function 
		'statesPerDataFrame' : 89,
		'n_components' : [3, 2],  # [gyro, accel]       # !!! note: changing this will change fig sizes 
        'butter_order' : 4,   # butterworth filter order
        'Fc' : 0.1,   # cutoff freq (Hz)
        'plot_flag_imu' : True  # !!! remove and just always save plots
}

cfg_motion_correct = {
    #'flag_do_splineSG' : False, # !!! This is not doing anything. left out for now. if True, will do splineSG motion correction
    #'splineSG_p' : 0.99, 
    #'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,  
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : cfg_imu_glm,
}

cfg_bandpass = { 
    'flag_bandpass_filter' : True,
    'fmin' : 0.01 * units.Hz, #0.02 * units.Hz,
    'fmax' : 0.5 * units.Hz  #3 * units.Hz
}


cfg_GLM = {    # this is a "filter," - we are getting HRFs from block average
    'drift_order' : 1,
    'distance_threshold' : 20 *units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : "ols",    # !!! add choice of basis func 
    't_delta' : 1 *units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1 *units.s ,     #  the temporal spacing between consecutive gaussians
    'cfg_hrf' : cfg_hrf
    }           


cfg_preprocess = {
    'flag_prune_channels' : False,  # FALSE = does not prune chans and does weighted averaging, TRUE = prunes channels and no weighted averaging
    'flag_do_GLM_filter' : True,
    'flag_save_preprocessed_data' : True,
    'median_filt' : 1, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'cfg_GLM' : cfg_GLM 
}


cfg_mse_conc = {                
    'mse_val_for_bad_data' : 1e7 * units.micromolar**2, 
    'mse_amp_thresh' : 1.1e-6,
    'mse_min_thresh' : 1e0 * units.micromolar**2, 
    'blockaverage_val' : 0 * units.micromolar
    }

# if block averaging on OD:
cfg_mse_od = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1.1e-6,
    'mse_min_thresh' : 1e-6,
    'blockaverage_val' : 0      # blockaverage val for bad data?
    }

cfg_blockavg = {
    'rec_str' : 'od_corrected',   # what you want to block average (will be either 'od_corrected' or 'conc')
    'flag_prune_channels' : cfg_preprocess['flag_prune_channels'],
    'cfg_hrf' : cfg_hrf,
    'trange_hrf_stat' : [10, 20],  
    'flag_save_block_avg_hrf': True,
    'flag_save_each_subj' : False,  # !!! do we need this?  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               


save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')



#%%

# File naming stuff
p_save_str = ''
if cfg_motion_correct['flag_do_imu_glm']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_imuGLM' 
else:
    p_save_str =  p_save_str
if cfg_motion_correct['flag_do_tddr']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_tddr' 
else:
    p_save_str =  p_save_str 
if cfg_preprocess['flag_do_GLM_filter']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_GLMfilt' 
else:
    p_save_str =  p_save_str   
if cfg_preprocess['flag_prune_channels']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_pruned' 
else:
    p_save_str =  p_save_str + '_unpruned' 



# SAVE cfg params to json file
dict_cfg_save = {"cfg_hrf": cfg_hrf, "cfg_dataset" : cfg_dataset, "cfg_preprocess" : cfg_preprocess, "cfg_GLM" : cfg_GLM, "cfg_blockavg" : cfg_blockavg}

cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str 

save_json_path = os.path.join(save_path, cfg_save_str + '.json')
save_pickle_path = os.path.join(save_path, cfg_save_str + '.pkl')
        
    
with open(os.path.join(save_json_path), "w", encoding="utf-8") as f:
    json.dump(dict_cfg_save, f, indent=4, default = str)  # Save as JSON with indentation

# Save configs as Pickle for Python usage (preserving complex objects like Pint quantities)
with open(save_pickle_path, "wb") as f:
    pickle.dump(dict_cfg_save, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Preprocessed data successfully saved.")


#%%  Try loading in pickle file and see wht happens

filepath_pkl = os.path.join(save_path, "cfg_params_STS_tddr_GLMfilt_unpruned.pkl")

if os.path.exists(filepath_pkl):
    with open(filepath_pkl, 'rb') as f:
        cfg = pickle.load(f)

cfg_hrf = cfg["cfg_hrf"]
cfg_dataset = cfg["cfg_dataset"]
cfg_preprocess = cfg["cfg_preprocess"]
cfg_GLM = cfg["cfg_GLM"]
cfg_blockavg = cfg["cfg_blockavg"]










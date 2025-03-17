#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing and block average script for fNIRS data analysis

Created on Tue Mar  4 11:22:37 2025

@author: smkelley
"""

# %% Imports
##############################################################################
#%matplotlib widget

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

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


# import my own functions from a different directory
import sys
#sys.path.append('/Users/dboas/Documents/GitHub/cedalion-dab-funcs')
#sys.path.append('C:\\Users\\shank\\Documents\\GitHub\\cedalion-dab-funcs2')
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2')
import DABfuncs_load_and_preprocess as pfDAB
import DABfuncs_plot_DQR as pfDAB_dqr
import DABfuncs_group_avg as pfDAB_grp_avg
import DABfuncs_ERBM_ICA as pfDAB_ERBM
import DABfuncs_image_recon as pfDAB_img
import spatial_basis_funs_ced as sbf 


# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

import pyvista as pv
# %% 
##############################################################################
import importlib
importlib.reload(pfDAB_dqr)
importlib.reload(pfDAB)
importlib.reload(pfDAB_grp_avg)


# %% Initial root directory and analysis parameters
##############################################################################


cfg_dataset = {
    #'root_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/Esplanade/',
    #'root_dir' : "D:\\fNIRS\\DATA\\Interactive_Walking_HD\\",
    'root_dir' : "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/",
    'subj_ids' : ['01','02','03','04','05','06','07','08','09','10', '11', '12', '13', '14'],
    #'subj_ids' : ['01'],
    'file_ids' : ['IWHD_run-01'],
    'subj_id_exclude' : ['10'], #['05','07'] # if you want to exclude a subject from the group average
    
    #'stim_lst' : ['ST', 'DT']  # FIXME: use this instead of having separate stim lists for hrf, dqr, ica, etc ???? - SK
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]

cfg_dqr = {
    'stim_lst_dqr' : ['ST', 'DT'] # FIXME: why have multiple of this
    #'stim_lst_dqr' : ['STS']
}

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

cfg_imu_glm = {'statesPerDataFrame' : 89,
		'hWin' : np.arange(-3,5,1), # window for impulse response function 
		'statesPerDataFrame' : 89,
		'n_components' : [3, 2],  # [gyro, accel]       # --- note: this will change fig sizes
        'butter_order' : 4,   # butterworth filter order
        'Fc' : 0.1,   # cutoff freq (Hz)
        'plot_flag_imu' : True
}

cfg_motion_correct = {
    'flag_do_splineSG' : False, # if True, will do splineSG motion correction
    'splineSG_p' : 0.99, 
    'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : cfg_imu_glm,
}

cfg_bandpass = { 
    'fmin' : 0.02 * units.Hz,
    'fmax' : 3 * units.Hz
}

cfg_preprocess = {
    'median_filt' : 3, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass
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


cfg_GLM = {
    'drift_order' : 1,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : "ols",
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,     #  the temporal spacing between consecutive gaussians
    }


cfg_blockavg = {
    'rec_str_lst' : ['od_tddr', 'od_o_tddr', 'od_imu_tddr', 'od_o_imu_tddr'],
    #'rec_str_lst' : ['od_tddr', 'od_o_tddr'],   # list of rec_str you want to block average
    'rec_str_lst_use_weighted' : [False, True, False, True] ,  # list indicating whether to save weighted for each rec_str
    'trange_hrf' : [5, 35] * units.s,
    'trange_hrf_stat' : [10, 20],
    'stim_lst_hrf' : ['ST', 'DT'], # FIXME: why have multiple of this
    #'stim_lst_hrf' : ['STS'], # FIXME: why have multiple of this
    'flag_do_GLM_filter' : True,
    'cfg_GLM' : cfg_GLM,
    'flag_save_group_avg_hrf': True,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               # !!! provide list of rec str and whether or not to save weighted for each one


cfg_erbmICA = {}

save_path = cfg_dataset['root_dir'] + 'derivatives/processed_data/'

flag_load_preprocessed_data = True  # if 1, will skip load_and_preprocess function and use saved data
flag_save_preprocessed_data = False   # SAVE or no save

flag_load_blockaveraged_data = False


# %% Load and preprocess the data
##############################################################################

# determine the number of subjects and files. Often used in loops.
n_subjects = len(cfg_dataset['subj_ids'])
n_files_per_subject = len(cfg_dataset['file_ids'])

# files to load
for subj_id in cfg_dataset['subj_ids']:
    subj_idx = cfg_dataset['subj_ids'].index(subj_id)
    for file_id in cfg_dataset['file_ids']:
        file_idx = cfg_dataset['file_ids'].index(file_id)
        filenm = f'sub-{subj_id}_task-{file_id}_nirs'
        if subj_idx == 0 and file_idx == 0:
            cfg_dataset['filenm_lst'] = []
            cfg_dataset['filenm_lst'].append( [filenm] )
        elif file_idx == 0:
            cfg_dataset['filenm_lst'].append( [filenm] )
        else:
            cfg_dataset['filenm_lst'][subj_idx].append( filenm )


import importlib
importlib.reload(pfDAB)


# RUN PREPROCESSING
if not flag_load_preprocessed_data:
    print("Running load and process function")
    rec, chs_pruned_subjs = pfDAB.load_and_preprocess( cfg_dataset, cfg_preprocess, cfg_dqr )
    
    
    # SAVE preprocessed data 
    if flag_save_preprocessed_data:
    
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0]+ '.pkl'), 'wb') as f: # !!! FIX ME: naming convention assumes file_ids only includes ONE task
            pickle.dump(chs_pruned_subjs, f, protocol=pickle.HIGHEST_PROTOCOL )
            
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + '.pkl'), 'wb') as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL )
            
            
        # SAVE cfg params to json file
        # !!! ADD image recon cfg 
        dict_cfg_save = {"cfg_dataset" : cfg_dataset, "cfg_preprocess" : cfg_preprocess, "cfg_GLM" : cfg_GLM, "cfg_blockavg" : cfg_blockavg}
        cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + '.json'

        with open(os.path.join(save_path, cfg_save_str), "w", encoding="utf-8") as f:
            json.dump(dict_cfg_save, f, indent=4, default = str)  # Save as JSON with indentation
        
        
# LOAD in saved data
else:
    print("Loading saved data")
    with gzip.open( os.path.join(save_path, 'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + '.pkl'), 'rb') as f: # !!! FIX ME: this assumes file_ids only includes ONE task
         rec = pickle.load(f)
    with gzip.open( os.path.join(save_path, 'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + '.pkl'), 'rb') as f:
         chs_pruned_subjs = pickle.load(f)



# %% Block Average - unweighted and weighted
##############################################################################


flag_load_blockaveraged_data = True

rec_str_lst = ['od_tddr', 'od_o_tddr', 'od_imu_tddr', 'od_o_imu_tddr']

rec_str_lst_use_weighted = [False, True, False, True] 
# !!!  ^^^use for  image recon -> if Cmeas false or true

# for saving file name 
if 'conc' in rec_str_lst[0]:  
    save_str = 'CONC'
else:
    save_str = 'OD'

# Compute block average
if not flag_load_blockaveraged_data:
    # !!! ADD in function if rec_str_use_weighted_lst is NONE then assume to save weighted for all trial types and for new_lst_use_weighted return NONE
    # !!! ^^ OR ASSUME SAVE UNWEIGHTED???
    blockaverage_mean, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.get_group_avg_for_diff_conds(rec, 
                                                                                                                                 rec_str_lst, rec_str_lst_use_weighted,  chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # save the results to a pickle file
    blockaverage = blockaverage_mean

    # Compute new rec_str_lst_use_weighted
    new_rec_str_lst_use_weighted = []
    [new_rec_str_lst_use_weighted .extend([value] * len(cfg_blockavg['stim_lst_hrf'])) for value in rec_str_lst_use_weighted] # Repeat each value for each trial type

    if cfg_blockavg['flag_save_each_subj']:
        # FIXME: this assumes the number of subjects and trial_type. Generalize this in the future.
        # blockaverage = blockaverage.sel(trial_type=['ST', 'ST-ica', 'ST-01', 'ST-ica-01', 'ST-02', 'ST-ica-02', 'ST-03', 'ST-ica-03', 'ST-04', 'ST-ica-04', 'ST-05', 'ST-ica-05', 'ST-06', 'ST-ica-06', 'ST-07', 'ST-ica-07', 'ST-08', 'ST-ica-08', 'ST-09', 'ST-ica-09'])
        blockaverage = blockaverage.sel(trial_type=['ST', 'ST-ica', 'ST-01', 'ST-ica-01', 'ST-02', 'ST-ica-02', 'ST-03', 'ST-ica-03', 'ST-04', 'ST-ica-04', 'ST-06', 'ST-ica-06', 'ST-08', 'ST-ica-08', 'ST-09', 'ST-ica-09'])
    
    
    groupavg_results = {'blockaverage': blockaverage_mean,
               'blockaverage_stderr': blockaverage_stderr,
               'blockaverage_subj': blockaverage_subj,
               'blockaverage_mse_subj': blockaverage_mse_subj,
               'new_rec_str_lst_use_weighted' : new_rec_str_lst_use_weighted,
               'geo2d' : rec[0][0].geo2d,
               'geo3d' : rec[0][0].geo3d
               }
    
    if cfg_blockavg['flag_save_group_avg_hrf']:
        file_path_pkl = os.path.join(save_path, 'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + '_' + save_str + '.pkl.gz')
        file = gzip.GzipFile(file_path_pkl, 'wb')
        file.write(pickle.dumps(groupavg_results))
        file.close()
        print('Saved group average HRF to ' + file_path_pkl)

else: # LOAD data
    filname =  'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + '_' + save_str + '.pkl.gz'
    filepath_bl = os.path.join(save_path , filname)
    
    if os.path.exists(filepath_bl):
        with gzip.open(filepath_bl, 'rb') as f:
            groupavg_results = pickle.load(f)
        blockaverage = groupavg_results['blockaverage']
        blockaverage_stderr = groupavg_results['blockaverage_stderr']
        blockaverage_subj = groupavg_results['blockaverage_subj']
        blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
        new_rec_str_lst_use_weighted = groupavg_results['new_rec_str_lst_use_weighted']
        geo2d = groupavg_results['geo2d']
        geo2d = groupavg_results['geo3d']
        print("Blockaverage file loaded successfully!")
    
    else:
        print(f"Error: File '{filepath_bl}' not found!")
        
blockaverage_all = blockaverage.copy()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 13:17:45 2025

@author: smkelley
"""
#%% Import modules
import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np

import cedalion
import cedalion_parcellation.datasets as datasets
import cedalion_parcellation.imagereco.forward_model as fw
import cedalion.io as io
from cedalion import units
import cedalion.dataclasses as cdc 

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import sys
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2/modules')
import module_image_recon as img_recon 
import module_spatial_basis_funs_ced as sbf 


#%% Load in cfg pickle files

#cfg_pkl_name = "cfg_params_STS_tddr_GLMfilt_unpruned.pkl" # STS
cfg_pkl_name = "cfg_params_IWHD_imuGLM_tddr_GLMfilt_unpruned.pkl"  # IWHD

cfg_filepath = os.path.join("/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/processed_data/",  # CHANGE
                            cfg_pkl_name) 

with open(cfg_filepath, 'rb') as file: # Open the file in binary read mode and load its contents
    cfg_params = pickle.load(file)

cfg_hrf = cfg_params["cfg_hrf"]
cfg_dataset = cfg_params["cfg_dataset"]
cfg_GLM = cfg_params["cfg_GLM"]
cfg_preprocess = cfg_params["cfg_preprocess"]
cfg_blockavg = cfg_params["cfg_blockavg"]
cfg_motion_correct = cfg_preprocess["cfg_motion_correct"]

subj_ids_new = [s for s in cfg_dataset['subj_ids'] if s not in cfg_dataset['subj_id_exclude']]


# Load in image recon cfg
cfg_pkl_name = "cfg_params_IWHD_imuGLM_tddr_GLMfilt_unpruned_cov_alpha_spatial_1e-02_alpha_meas_1e-02_indirect_Cmeas_SB.pkl" # CHANGE
cfg_filepath = os.path.join("/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/processed_data/image_recon/",  # CHANGE
                            cfg_pkl_name) 

with open(cfg_filepath, 'rb') as file:  # Open the file in binary read mode and load its contents
    cfg_params = pickle.load(file)

cfg_sb = cfg_params["cfg_sb"]
cfg_img_recon = cfg_params["cfg_img_recon"]


save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 'image_recon')


# File naming stuff
if cfg_img_recon['DIRECT']:
    direct_name = 'direct'
else:
    direct_name = 'indirect'
    
if cfg_img_recon['SB']:
    SB_name = 'SB'
else:
    SB_name = 'noSB'

if cfg_img_recon['flag_Cmeas']:
    Cmeas_name = 'Cmeas'
else:
    Cmeas_name = 'noCmeas'


#%% Load image recon results

filname = 'Xs_IWHD_cov_alpha_spatial_1e-02_alpha_meas_1e-02_indirect_Cmeas_SB.pkl.gz' # CHANGE

filepath_bl = os.path.join(save_path , filname)

if os.path.exists(filepath_bl):
    with gzip.open(filepath_bl, 'rb') as f:
        results = pickle.load(f)
        
    all_trial_X_hrf_mag = results["X_hrf_mag"]
    all_trial_X_hrf_mag_weighted = results["X_hrf_mag_weighted"]
    all_trial_X_stderr = results["X_std_err"]
    all_trial_X_tstat = results["X_tstat"]
    all_trial_X_mse_between = results["X_mse_between"]
    all_trial_X_mse_within = results["X_mse_within"]
    
    print("Image results file loaded successfully!")
    
else:
    print(f'{filepath_bl} does not exist')

#%% load head model 
#probe_dir = "/projectnb/nphfnirs/ns/Shannon/Data/probes/NN22_WHHD/12NN/" 
import importlib
importlib.reload(img_recon)

head, PARCEL_DIR = img_recon.load_head_model(cfg_img_recon['head_model'], with_parcels=False)
Adot, meas_list, geo3d, amp = img_recon.load_probe(cfg_img_recon['probe_dir'], snirf_name='fullhead_56x144_NN22_System1.snirf')


ec = cedalion.nirs.get_extinction_coefficients('prahl', Adot.wavelength)
einv = cedalion.xrutils.pinv(ec)

#%% build plots 
import importlib
importlib.reload(img_recon)

threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
SAVE = True
flag_hbo_list = [True, False]
flag_brain_list = [True, False]
flag_img_list = ['mag', 'tstat', 'noise'] #, 'noise'

if cfg_dataset["file_ids"][0].split("_")[0] == 'IWHD':
    flag_condition_list = ['ST', 'DT']   
elif cfg_dataset["file_ids"][0].split("_")[0] == 'STS':
    flag_condition_list = ['STS'] 

# with gzip.open( filepath, 'rb') as f:
#      results = pickle.load(f)

# all_trial_X_hrf_mag = results['X_hrf_mag']
for flag_hbo in flag_hbo_list:
    
    for flag_brain in flag_brain_list: 
        
        for flag_condition in flag_condition_list:
            
            for flag_img in flag_img_list:
                
                if flag_hbo:
                    title_str = flag_condition + ' ' + 'HbO'
                    hbx_brain_scalp = 'hbo'
                else:
                    title_str = flag_condition + ' ' + 'HbR'
                    hbx_brain_scalp = 'hbr'
                
                if flag_brain:
                    title_str = title_str + ' brain'
                    hbx_brain_scalp = hbx_brain_scalp + '_brain'
                else:
                    title_str = title_str + ' scalp'
                    hbx_brain_scalp = hbx_brain_scalp + '_scalp'
                
                if len(flag_condition_list) > 1:
                    if flag_img == 'tstat':
                        foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                        title_str = title_str + ' t-stat'
                    elif flag_img == 'mag':
                        foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()
                        title_str = title_str + ' magnitude'
                    elif flag_img == 'noise':
                        foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                        title_str = title_str + ' noise'
                else:
                    if flag_img == 'tstat':
                        foo_img = all_trial_X_tstat.copy()
                        title_str = title_str + ' t-stat'
                    elif flag_img == 'mag':
                        foo_img = all_trial_X_hrf_mag_weighted.copy()
                        title_str = title_str + ' magnitude'
                    elif flag_img == 'noise':
                        foo_img = all_trial_X_stderr.copy()
                        title_str = title_str + ' noise'
        
                foo_img = foo_img.pint.dequantify()
                foo_img = foo_img.transpose('vertex', 'chromo')
                foo_img[~M] = np.nan
                
             # 
                clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
                # if flag_img == 'mag':
                #     clim = [-7.6e-4, 7.6e-4]
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                                          None, title_str, off_screen=SAVE )
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
                
                if SAVE:
                    img_folder = f'{direct_name}_aspatial-{cfg_img_recon["alpha_spatial"]}_ameas-{cfg_img_recon["alpha_meas"]}_{Cmeas_name}_{SB_name}'
                    save_dir_tmp= os.path.join(cfg_dataset["root_dir"], 'derivatives', 'plots', 'image_recon', img_folder)
                    if not os.path.exists(save_dir_tmp):
                        os.makedirs(save_dir_tmp)
                    file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
                    p0.screenshot( os.path.join(save_dir_tmp, file_name) )
                    p0.close()
                else:
                    p0.show()
                    

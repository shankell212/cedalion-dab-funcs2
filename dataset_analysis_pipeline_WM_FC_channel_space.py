# %% Imports
##############################################################################
#%matplotlib widget



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


# import my own functions from a different directory
import sys
sys.path.append('/Users/dboas/Documents/GitHub/cedalion-dab-funcs/modules')
#sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2/modules')
import module_load_and_preprocess as pfDAB
import module_plot_DQR as pfDAB_dqr
import module_group_avg as pfDAB_grp_avg    
import module_ERBM_ICA as pfDAB_ERBM
import module_image_recon as pfDAB_img
import module_spatial_basis_funs_ced as sbf 



# %% Initial root directory and analysis parameters
##############################################################################


cfg_hrf = {
    'stim_lst' : ['ST', 'DT'], 
    't_pre' : 5 *units.s, 
    't_post' : 33 *units.s
    #'t_post' : [ 33, 33 ] *units.s   # !!! GLM does not let you have different time ranges for diff stims right now
    }

cfg_dataset = {
    'root_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/Laura_WMandRest/',
    'subj_ids' : ['01','02','03','04','06','07','08'], # '05' removed for now because of NaN after tddr()
    'file_ids' : ['WM_run-01','WM_run-02','WM_run-03','WM_run-04'],
    'subj_id_exclude' : [], #['05','07'] # if you want to exclude a subject from the group average
    'cfg_hrf' : cfg_hrf,
    'derivatives_subfolder' : ''    
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
		'n_components' : [3, 2],  # [gyro, accel]       # !!! note: changing this will change fig sizes - add that in?
        'butter_order' : 4,   # butterworth filter order
        'Fc' : 0.1,   # cutoff freq (Hz)
        'plot_flag_imu' : True  
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
    'fmin' : 0.01 * units.Hz, #0.02 * units.Hz,
    'fmax' : 0.5 * units.Hz,  #3 * units.Hz
    'flag_bandpass_filter' : True
}


cfg_GLM = {
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
    'median_filt' : 3, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'flag_do_GLM_filter' : False,
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
    'mse_min_thresh' : 1e-6,  # LC using 1e-3 ?
    'blockaverage_val' : 0      # blockaverage val for bad data?
    }

cfg_blockavg = {
    'rec_str' : 'od_corrected',   # what you want to block average (will be either 'od_corrected' or 'conc')
    'flag_prune_channels' : cfg_preprocess['flag_prune_channels'],
    'cfg_hrf' : cfg_hrf,
    'trange_hrf_stat' : [10, 20],  
    'flag_save_group_avg_hrf': False,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               



cfg_erbmICA = {}

save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')

flag_load_preprocessed_data = True  
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
    
    
# RUN PREPROCESSING
if not flag_load_preprocessed_data:
    print("Running load and process function")
    
    # RUN preprocessing
    rec, chs_pruned_subjs = pfDAB.load_and_preprocess( cfg_dataset, cfg_preprocess ) 

    
    # SAVE preprocessed data 
    if flag_save_preprocessed_data:
        print(f"Saving preprocessed data for {cfg_dataset['file_ids']}")
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'wb') as f: # !!! FIX ME: naming convention assumes file_ids only includes ONE task
            pickle.dump(chs_pruned_subjs, f, protocol=pickle.HIGHEST_PROTOCOL )
            
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'wb') as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL )
            
            
        # SAVE cfg params to json file
        # !!! ADD image recon cfg  ?? - or make it its own .json since i am planning to separate into 2 scripts
        dict_cfg_save = {"cfg_hrf": cfg_hrf, "cfg_dataset" : cfg_dataset, "cfg_preprocess" : cfg_preprocess, "cfg_GLM" : cfg_GLM, "cfg_blockavg" : cfg_blockavg}
        
        cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.json'
            
        with open(os.path.join(save_path, cfg_save_str), "w", encoding="utf-8") as f:
            json.dump(dict_cfg_save, f, indent=4, default = str)  # Save as JSON with indentation
        print("Preprocessed data successfully saved.")
        
        
# LOAD IN SAVED DATA
else:
    print("Loading saved data")   # !!! update with new naming for pruned or unpruned above
    with gzip.open( os.path.join(save_path, 'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'rb') as f: # !!! FIX ME: this assumes file_ids only includes ONE task
         rec = pickle.load(f)
    with gzip.open( os.path.join(save_path, 'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'rb') as f:
         chs_pruned_subjs = pickle.load(f)
    print(f'Data loaded successfully for {cfg_dataset["file_ids"][0].split("_")[0]}')









# %% Functional Connectivity
# Preprocess the data
##############################################################################
import importlib
importlib.reload(pfDAB_img)

flag_do_bp_filter_on_od = True
flag_do_gms_chromo = True

flag_include_full_ts = True

flag_channels_to_parcels = True
flag_parcels_use_lev1 = True # if False then use Lev2

unique_trial_types = np.unique(rec[0][0].stim.trial_type) # we assume all subjects have the same trial types
if flag_include_full_ts:
    unique_trial_types = np.append(unique_trial_types, 'full_ts')

# Loop over subjects
for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

    # if idx_subj > 0:
    #     break

    print(f'Preprocess SUBJECT {curr_subj}')

    conc_ts_files = {}
    conc_var_files = {}

    # Initialize the time series for each subject 
    if idx_subj == 0:
        conc_ts_subjs = {}
        conc_var_subjs = {}
    conc_ts_subjs[idx_subj] = {}
    conc_var_subjs[idx_subj] = {}

    corr_hbo_files = {}
    corr_hbr_files = {}

    # Loop over runs
    for idx_file, curr_file in enumerate(cfg_dataset['file_ids']):

        # if idx_file >0:
        #     break

        # get the OD time series
        od_ts = rec[idx_subj][idx_file]['od_corrected'].copy()

        # bandpass filter the parcel time series
        if flag_do_bp_filter_on_od:
            fmin = 0.01 * units.Hz
            fmax = 0.2 * units.Hz
            od_ts = cedalion.sigproc.frequency.freq_filter(od_ts, fmin, fmax)

        # convert to conc
        dpf = xr.DataArray(
            [1, 1],
            dims="wavelength",
            coords={"wavelength": od_ts.wavelength},
        )
        conc_ts = cedalion.nirs.od2conc(od_ts, rec[idx_subj][idx_file].geo3d, dpf, spectrum="prahl")

        # Global mean subtraction for each chromo
        if flag_do_gms_chromo:
            # do a weighted mean by variance
            conc_var = conc_ts.var('time')

            # correct for bad data
            amp = rec[idx_subj][idx_file]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
            idx_amp = np.where(amp < cfg_blockavg['cfg_mse_od']['mse_amp_thresh'])[0]
            conc_var.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
            conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

            idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0] 
            conc_var.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
            conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

            # FIXME: deal with rare instances when conc_var is 0

            gms = (conc_ts / conc_var).mean('channel') / (1/conc_var).mean('channel')
            numerator = (conc_ts * gms).sum(dim="time")
            denominator = (gms * gms).sum(dim="time")
            scl = numerator / denominator
            conc_ts = conc_ts - scl*gms

        # loop over trial types
        for trial_type in unique_trial_types:
            if trial_type != 'full_ts':
                idx = np.where(rec[idx_subj][idx_file].stim.trial_type==trial_type)[0]
                t_indices_tmp = np.array([])
                dt = np.median(np.diff(conc_ts.time)) 
                for ii in idx:
                    t_indices_tmp = np.concatenate( (t_indices_tmp, np.where( 
                                        (conc_ts.time >  rec[idx_subj][idx_file].stim.onset[ii]) &
                                        (conc_ts.time <= (rec[idx_subj][idx_file].stim.onset[ii] + np.floor(rec[idx_subj][idx_file].stim.duration[ii]/dt)*dt + 1e-4 )) # this dt stuff is to ensure same lengths for each trial_type, BUT not needed anymore
                                        )[0] )
                                    )
                foo_ts = conc_ts.isel(time=t_indices_tmp.astype(int)).copy() 
            else:
                # get the full time series
                foo_ts = conc_ts.copy()

            # get the correlation matrix for assessing repeatability
            corr_hbo = np.corrcoef( foo_ts.sel(chromo='HbO').values, rowvar=True )
            corr_hbr = np.corrcoef( foo_ts.sel(chromo='HbR').values, rowvar=True )

            # store results 
            # concatenate the time series and variance for each trial type for subsequent corrcoef in next cell
            # store corrcoef for each file for repeatability
            if idx_file == 0:
                conc_ts_files[trial_type] = foo_ts
                conc_var_files[trial_type] = conc_var.copy()
                corr_hbo_files[trial_type] = np.zeros((len(cfg_dataset['file_ids']), len(conc_ts.channel)**2))
                corr_hbr_files[trial_type] = np.zeros((len(cfg_dataset['file_ids']), len(conc_ts.channel)**2))
                corr_hbo_files[trial_type][idx_file, :] = corr_hbo.reshape(-1)
                corr_hbr_files[trial_type][idx_file, :] = corr_hbr.reshape(-1)
            else:
                conc_ts_files[trial_type] = xr.concat([conc_ts_files[trial_type], foo_ts], dim='time', coords='minimal', compat='override') # ensure no reordering since times overlap
                conc_var_files[trial_type] = xr.concat([conc_var_files[trial_type], conc_var], dim='file')
                corr_hbo_files[trial_type][idx_file, :] = corr_hbo.reshape(-1)
                corr_hbr_files[trial_type][idx_file, :] = corr_hbr.reshape(-1)
        # end of trial type loop

    # end of file loop

    # store the time series for each subject for each trial_type
    for trial_type in unique_trial_types:
        conc_ts_subjs[idx_subj][trial_type] = conc_ts_files[trial_type].copy()
        conc_var_subjs[idx_subj][trial_type] = conc_var_files[trial_type].mean('file') 

        if flag_channels_to_parcels: # project channels to parcel_lev2 by weighted average over channels
            w = 1 / conc_var_subjs[idx_subj][trial_type]
            # get the normalized weighted averaging kernel
            if flag_parcels_use_lev1:
                Adot_parcels_weighted_xr = w * Adot_parcels_lev1_xr
            else:
                Adot_parcels_weighted_xr = w * Adot_parcels_lev2_xr
            Adot_parcels_weighted_xr = Adot_parcels_weighted_xr / Adot_parcels_weighted_xr.sum(dim='channel')
            # do the inner product over channel between conc_ts_subjs[idx_subj][trial_type] and Adot_parcels_lev2_weighted_xr
            foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T @ conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO')
            foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T @ conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR')
            conc_ts_subjs[idx_subj][trial_type] = xr.concat([foo_hbo, foo_hbr], dim='chromo')
            # do the same with conc_var_subjs[idx_subj][trial_type]
            foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T**2 @ conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO')
            foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T**2 @ conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR')
            conc_var_subjs[idx_subj][trial_type] = xr.concat([foo_hbo, foo_hbr], dim='chromo')


    # get the repeatability
    if idx_subj == 0:
        repeatability_subj_hbo_mean = {}
        repeatability_subj_hbo_std = {}
        repeatability_subj_hbr_mean = {}
        repeatability_subj_hbr_std = {}
    for trial_type in unique_trial_types:
        foo_hbo = np.corrcoef(np.nan_to_num(corr_hbo_files[trial_type]), rowvar=True)
        foo_hbr = np.corrcoef(np.nan_to_num(corr_hbr_files[trial_type]), rowvar=True)
        if idx_subj==0:
            repeatability_subj_hbo_mean[trial_type] = np.zeros(n_subjects)
            repeatability_subj_hbo_std[trial_type] = np.zeros(n_subjects)
            repeatability_subj_hbr_mean[trial_type] = np.zeros(n_subjects)
            repeatability_subj_hbr_std[trial_type] = np.zeros(n_subjects)
        repeatability_subj_hbo_mean[trial_type][idx_subj] = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].mean()
        repeatability_subj_hbo_std[trial_type][idx_subj] = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].std()
        repeatability_subj_hbr_mean[trial_type][idx_subj] = foo_hbr[np.triu_indices(foo_hbo.shape[0], k=1)].mean()
        repeatability_subj_hbr_std[trial_type][idx_subj] = foo_hbr[np.triu_indices(foo_hbo.shape[0], k=1)].std()
# end of subject loop






# %%
# Get the correlation matrices

corr_hbo_subj = {}
corr_hbr_subj = {}
corr_hbo_subj_var = {}
corr_hbr_subj_var = {}

# Loop over subjects
for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

    # if idx_subj > 0:
    #     break

    print(f'Correlation Matrices for SUBJECT {curr_subj}')

    # loop over trial types
    for trial_type in unique_trial_types:

        # get the correlation matrix
        corr_hbo = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values, rowvar=True )
        corr_hbr = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values, rowvar=True )

        # get the correlation matrix for each subject
        if idx_subj == 0:
            corr_hbo_subj[trial_type] = np.zeros((len(cfg_dataset['subj_ids']), corr_hbo.shape[0]*corr_hbo.shape[1]))
            corr_hbr_subj[trial_type] = np.zeros((len(cfg_dataset['subj_ids']), corr_hbo.shape[0]*corr_hbo.shape[1]))
            corr_hbo_subj_var[trial_type] = np.zeros((len(cfg_dataset['subj_ids']), corr_hbo.shape[0]*corr_hbo.shape[1]))
            corr_hbr_subj_var[trial_type] = np.zeros((len(cfg_dataset['subj_ids']), corr_hbo.shape[0]*corr_hbo.shape[1]))
        corr_hbo_subj[trial_type][idx_subj, :] = corr_hbo.reshape(-1)
        corr_hbr_subj[trial_type][idx_subj, :] = corr_hbr.reshape(-1)
        # get the variance of each element in the correlation matrix
        corr_hbo_subj_var[trial_type][idx_subj, :] = (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis].T).reshape(-1)
        corr_hbr_subj_var[trial_type][idx_subj, :] = (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis].T).reshape(-1)
    # end of trial type loop
# end of subject loop

# get the reliability
reliability_hbo_mean = {}
reliability_hbo_std = {}
reliability_hbr_mean = {}
reliability_hbr_std = {}
for trial_type in unique_trial_types:
    foo_hbo = np.corrcoef(np.nan_to_num(corr_hbo_subj[trial_type]), rowvar=True)
    foo_hbr = np.corrcoef(np.nan_to_num(corr_hbr_subj[trial_type]), rowvar=True)
    reliability_hbo_mean[trial_type] = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].mean()
    reliability_hbo_std[trial_type] = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].std()
    reliability_hbr_mean[trial_type] = foo_hbr[np.triu_indices(foo_hbo.shape[0], k=1)].mean()
    reliability_hbr_std[trial_type] = foo_hbr[np.triu_indices(foo_hbo.shape[0], k=1)].std()



# %%
# print the repeatability and reliability
for trial_type in unique_trial_types:
    print(f"\nRepeatability '{trial_type}'")
    for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):
        print(f"Subject {curr_subj} repeatability HbO {repeatability_subj_hbo_mean[trial_type][idx_subj]:.2f} +/- {repeatability_subj_hbo_std[trial_type][idx_subj]:.2f}; HbR {repeatability_subj_hbr_mean[trial_type][idx_subj]:.2f} +/- {repeatability_subj_hbr_std[trial_type][idx_subj]:.2f}")
print('')
for trial_type in unique_trial_types:
    print(f"Reliability '{trial_type}' HbO {reliability_hbo_mean[trial_type]:.2f} +/- {reliability_hbo_std[trial_type]:.2f}; HbR {reliability_hbr_mean[trial_type]:.2f} +/- {reliability_hbr_std[trial_type]:.2f}")





# z-transform the correlation matrix across subjects using Fisher's z-transform
corrz_hbo_subj_mean = {}
corrz_hbr_subj_mean = {}
corrz_hbo_subj_std = {}
corrz_hbr_subj_std = {}
corrz_hbo_subj_meanweighted = {}
corrz_hbr_subj_meanweighted = {}
corr_hbo_subj_mean = {}
corr_hbr_subj_mean = {}
corr_hbo_subj_std = {}
corr_hbr_subj_std = {}
corr_hbo_subj_meanweighted = {}
corr_hbr_subj_meanweighted = {}

for trial_type in unique_trial_types:
    corrz_hbo_subj = np.arctanh(corr_hbo_subj[trial_type])
    corrz_hbr_subj = np.arctanh(corr_hbr_subj[trial_type])

    # get the mean and std of the correlation matrix across subjects
    corrz_hbo_subj_mean[trial_type] = np.nanmean(corrz_hbo_subj, axis=0)
    corrz_hbr_subj_mean[trial_type] = np.nanmean(corrz_hbr_subj, axis=0)
    corrz_hbo_subj_std[trial_type] = np.nanstd(corrz_hbo_subj, axis=0)
    corrz_hbr_subj_std[trial_type] = np.nanstd(corrz_hbr_subj, axis=0)

    corrz_hbo_subj_meanweighted[trial_type] = np.nansum(corrz_hbo_subj * corr_hbo_subj_var[trial_type], axis=0) / np.nansum( corr_hbo_subj_var[trial_type], axis=0)
    corrz_hbr_subj_meanweighted[trial_type] = np.nansum(corrz_hbr_subj * corr_hbr_subj_var[trial_type], axis=0) / np.nansum( corr_hbr_subj_var[trial_type], axis=0)

    # get the mean and std of the correlation matrix across subjects
    corr_hbo_subj_mean[trial_type] = np.tanh(corrz_hbo_subj_mean[trial_type])
    corr_hbr_subj_mean[trial_type] = np.tanh(corrz_hbr_subj_mean[trial_type])
    corr_hbo_subj_std[trial_type] = np.tanh(corrz_hbo_subj_std[trial_type])
    corr_hbr_subj_std[trial_type] = np.tanh(corrz_hbr_subj_std[trial_type])

    corr_hbo_subj_meanweighted[trial_type] = np.tanh(corrz_hbo_subj_meanweighted[trial_type])
    corr_hbr_subj_meanweighted[trial_type] = np.tanh(corrz_hbr_subj_meanweighted[trial_type])




# %%
# Boot Strap
# get the weighted mean of the correlation matrix and bootstrap to get the std

#trial_type_list = np.array(['full_ts']) # choose the trial type to use for bootstrapping
                       # 'active', 'passive', 'full_ts', 'active-passive'
trial_type_list = unique_trial_types
# append 'active-passive' to the list of trial types
if 'active-passive' not in trial_type_list:
    trial_type_list = np.append(trial_type_list, 'active-passive')

z_boot_mean_hbo = {}
z_boot_mean_hbr = {}
z_boot_se_hbo = {}
z_boot_se_hbr = {}
r_boot_mean_hbo = {}
r_boot_mean_hbr = {}

for trial_type in trial_type_list:

    print(f"Bootstrapping for trial type '{trial_type}'")

    n_boot = 1000  # number of bootstrap samples

    n_channels = int(np.sqrt(len(corr_hbo_subj['full_ts'][0])))

    z_boot_samples_hbo = np.zeros((n_boot, n_channels*n_channels))
    z_boot_samples_hbr = np.zeros((n_boot, n_channels*n_channels))

    for b in range(n_boot):
        # Resample subject indices with replacement
        boot_indices = np.random.choice(n_subjects, size=n_subjects, replace=True)

        z_weighted_sum_hbo = np.zeros((n_channels*n_channels))
        weight_sum_hbo = np.zeros((n_channels*n_channels))

        z_weighted_sum_hbr = np.zeros((n_channels*n_channels))
        weight_sum_hbr = np.zeros((n_channels*n_channels))

        for idx in boot_indices:
            if trial_type != 'active-passive':
                R = corr_hbo_subj[trial_type][idx]
                z = np.arctanh(R)  # Fisher z-transform
                w = 1 / corr_hbo_subj_var[trial_type][idx]
            else:
                z = np.arctanh(corr_hbo_subj['active'][idx] - corr_hbo_subj['passive'][idx])  # Fisher z-transform
                w = 1 / (corr_hbo_subj_var['active'][idx] + corr_hbo_subj_var['passive'][idx])
            z_weighted_sum_hbo += np.nan_to_num(z / w) # turn nan to 0
            weight_sum_hbo += np.nan_to_num(1 / w) # turn nan to 0

            if trial_type != 'active-passive':
                R = corr_hbr_subj[trial_type][idx]
                z = np.arctanh(R)  # Fisher z-transform
                w = 1 / corr_hbr_subj_var[trial_type][idx]
            else:
                z = np.arctanh(corr_hbr_subj['active'][idx] - corr_hbr_subj['passive'][idx])
                w = 1 / (corr_hbr_subj_var['active'][idx] + corr_hbr_subj_var['passive'][idx])
            z_weighted_sum_hbr += np.nan_to_num(z / w) # turn nan to 0
            weight_sum_hbr += np.nan_to_num(1 / w) # turn nan to 0

        z_boot_samples_hbo[b] = z_weighted_sum_hbo / weight_sum_hbo
        z_boot_samples_hbr[b] = z_weighted_sum_hbr / weight_sum_hbr

    # Mean of bootstrap samples
    z_boot_mean_hbo[trial_type] = np.mean(z_boot_samples_hbo, axis=0)
    z_boot_mean_hbr[trial_type] = np.mean(z_boot_samples_hbr, axis=0)

    # Standard error (SE)
    z_boot_se_hbo[trial_type] = np.std(z_boot_samples_hbo, axis=0)
    z_boot_se_hbr[trial_type] = np.std(z_boot_samples_hbr, axis=0)

    # Confidence intervals (e.g. 95%)
    z_ci_lower_hbo = np.percentile(z_boot_samples_hbo, 2.5, axis=0)
    z_ci_upper_hbo = np.percentile(z_boot_samples_hbo, 97.5, axis=0)
    z_ci_lower_hbr = np.percentile(z_boot_samples_hbr, 2.5, axis=0)
    z_ci_upper_hbr = np.percentile(z_boot_samples_hbr, 97.5, axis=0)

    # Convert back to r-space if needed
    r_boot_mean_hbo[trial_type] = np.tanh(z_boot_mean_hbo[trial_type])
    r_ci_lower_hbo = np.tanh(z_ci_lower_hbo)
    r_ci_upper_hbo = np.tanh(z_ci_upper_hbo)
    r_boot_mean_hbr[trial_type] = np.tanh(z_boot_mean_hbr[trial_type])
    r_ci_lower_hbr = np.tanh(z_ci_lower_hbr)
    r_ci_upper_hbr = np.tanh(z_ci_upper_hbr)







# %%
# plot the correlation matrix
trial_type = 'full_ts' 
#trial_type = 'active' 
#trial_type = 'passive'
#trial_type = 'active-passive'

vminmax = 1
#vminmax = 0.1

from scipy.stats import t
p_value = 1e-2
df = n_subjects-1  
t_crit = t.ppf(1 - p_value/2, df)  # For two-tailed test

#t_crit = 1

if 0: # mean across subjects
    print(f"t_crit: {t_crit:.2f}")
    f, axs = p.subplots(2,2,figsize=(12,12))

    ax1 = axs[0][0]
    foo1 = corr_hbo_subj_mean.copy()
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-1, vmax=1)
    ax1.set_title('HbO Correlation Matrix')

    ax1 = axs[0][1]
    foo1 = corr_hbr_subj_mean.copy()
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-1, vmax=1)
    ax1.set_title('HbR Correlation Matrix')

    ax1 = axs[1][0]
    foo = corr_hbo_subj_mean / (corr_hbo_subj_std / np.sqrt(n_subjects))
    foo1 = corr_hbo_subj_mean.copy()
    foo1[np.abs(foo) < t_crit] = np.nan # remove non-significant correlations
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-1, vmax=1)

    ax1 = axs[1][1]
    foo = corr_hbr_subj_mean / (corr_hbr_subj_std / np.sqrt(n_subjects))
    foo1 = corr_hbr_subj_mean.copy()
    foo1[np.abs(foo) < t_crit] = np.nan # remove non-significant correlations
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-1, vmax=1)

elif 1: # weighted mean across subjects
    print(f"t_crit: {t_crit:.2f}")
    f, axs = p.subplots(2,2,figsize=(12,8))

    ax1 = axs[0][0]
    foo1 = r_boot_mean_hbo[trial_type].copy()
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-vminmax, vmax=vminmax)
    ax1.set_title('HbO Correlation Matrix')

    ax1 = axs[0][1]
    foo1 = r_boot_mean_hbr[trial_type].copy()
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-vminmax, vmax=vminmax)
    ax1.set_title('HbR Correlation Matrix')

    ax1 = axs[1][0]
    foo = z_boot_mean_hbo[trial_type] / z_boot_se_hbo[trial_type]
    foo1 = r_boot_mean_hbo[trial_type].copy()
    foo1[np.abs(foo) < t_crit] = np.nan # remove non-significant correlations
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-vminmax, vmax=vminmax)

    ax1 = axs[1][1]
    foo = z_boot_mean_hbr[trial_type] / z_boot_se_hbr[trial_type]
    foo1 = r_boot_mean_hbr[trial_type].copy()
    foo1[np.abs(foo) < t_crit] = np.nan # remove non-significant correlations
    ax1.imshow( foo1.reshape((n_channels,n_channels)), cmap='jet', vmin=-vminmax, vmax=vminmax)

else: # each subject
    f, axs = p.subplots(n_subjects,2,figsize=(12,50))

    for ii in range(0, n_subjects):
        ax1 = axs[ii][0]
        ax1.imshow( corr_hbo_subj[ii,:].reshape((567,567)), cmap='jet', vmin=-1, vmax=1)

        ax1 = axs[ii][1]
        ax1.imshow( corr_hbr_subj[ii,:].reshape((567,567)), cmap='jet', vmin=-1, vmax=1)






# %%
#  mne_connectivity plot_connectivity_circle
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

# FIXME: can I get y pos of parcels and order by that?
# FIXME: would be nice to color code the 17 networks and visualize them on the brain. Maybe consider colors in examples at https://mne.tools/mne-connectivity/dev/generated/mne_connectivity.viz.plot_connectivity_circle.html


trial_type = 'full_ts' 
#trial_type = 'active' 
#trial_type = 'passive'
#trial_type = 'active-passive'

vminmax = (0.5,1)
#vminmax = (-0.1,0.1)
flag_show_tstat = False
flag_show_hbo = True # false means show HbR

#colormap = 'hot'
colormap = 'jet'

n_lines = 300

p_value = 1e-2
df = n_subjects-1  
t_crit = t.ppf(1 - p_value/2, df)  # For two-tailed test


flag_show_specific_parcels = True
#parcels_to_show = ['SomMotA_LH', 'SomMotA_RH']
parcels_to_show = ['DorsAttnA_LH', 'DorsAttnA_RH']
# parcels_to_show = ['SalVentAttnA_FrMed_LH', 'SalVentAttnA_FrMed_RH',
#        'SalVentAttnA_FrOper_LH', 'SalVentAttnA_FrOper_RH',
#        'SalVentAttnA_Ins_LH', 'SalVentAttnA_Ins_RH',
#        'SalVentAttnA_ParMed_LH', 'SalVentAttnA_ParMed_RH',
#        'SalVentAttnA_ParOper_LH', 'SalVentAttnA_ParOper_RH',
#        'SalVentAttnA_PrC_RH']
# parcels_to_show = ['SalVentAttnB_Cinga_RH',
#        'SalVentAttnB_IPL_RH', 'SalVentAttnB_Ins_LH',
#        'SalVentAttnB_Ins_RH', 'SalVentAttnB_PFCd_LH',
#        'SalVentAttnB_PFCd_RH', 'SalVentAttnB_PFCl_LH',
#        'SalVentAttnB_PFCl_RH', 'SalVentAttnB_PFClv_RH',
#        'SalVentAttnB_PFCmp_LH', 'SalVentAttnB_PFCmp_RH']
# parcels_to_show = ['SomMotB_Aud_LH', 'SomMotB_Aud_RH',
#        'SomMotB_Cent_LH', 'SomMotB_Cent_RH', 'SomMotB_Ins_LH',
#        'SomMotB_Ins_RH', 'SomMotB_S2_LH', 'SomMotB_S2_RH']
# parcels_to_show = ['DefaultA_IPL_LH', 'DefaultA_IPL_RH',
#        'DefaultA_PFCd_LH', 'DefaultA_PFCd_RH', 'DefaultA_PFCm_LH',
#        'DefaultA_PFCm_RH', 'DefaultA_Temp_RH', 'DefaultA_pCunPCC_LH',
#        'DefaultA_pCunPCC_RH']
# parcels_to_show = ['DefaultB_AntTemp_RH', 'DefaultB_IPL_LH',
#        'DefaultB_PFCd_LH', 'DefaultB_PFCd_RH', 'DefaultB_PFCl_LH',
#        'DefaultB_PFCv_LH', 'DefaultB_PFCv_RH', 'DefaultB_Temp_LH',
#        'DefaultB_Temp_RH']
# parcels_to_show = ['DefaultC_IPL_LH', 'DefaultC_IPL_RH',
#        'DefaultC_PHC_LH', 'DefaultC_PHC_RH', 'DefaultC_Rsp_LH',
#        'DefaultC_Rsp_RH']

# First, we reorder the labels based on their location in the left hemi
#label_names = [label.name for label in labels]
if flag_parcels_use_lev1:
    lh_labels = [name for name in unique_parcels_lev1 if name.endswith("LH")]
    rh_labels = [name for name in unique_parcels_lev1 if name.endswith("RH")] #[label[:-2] + "rh" for label in lh_labels]
    unique_parcels_list = unique_parcels_lev1.tolist()
else:
    lh_labels = [name for name in unique_parcels_lev2 if name.endswith("LH")]
    rh_labels = [name for name in unique_parcels_lev2 if name.endswith("RH")] #[label[:-2] + "rh" for label in lh_labels]
    unique_parcels_list = unique_parcels_lev2.tolist()

# Get the y-location of the label
# label_ypos = list()
# for name in lh_labels:
#     idx = label_names.index(name)
#     ypos = np.mean(labels[idx].pos[:, 1])
#     label_ypos.append(ypos)

# Reorder the labels based on their location
#lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]


# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels)  # reverse the order
node_order.extend(rh_labels[::-1])

# node_angles = circular_layout(
#     label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2]
# )




if flag_show_hbo:
    foo = z_boot_mean_hbo[trial_type] / z_boot_se_hbo[trial_type]
    foo1 = r_boot_mean_hbo[trial_type].copy()
else:
    foo = z_boot_mean_hbr[trial_type] / z_boot_se_hbr[trial_type]
    foo1 = r_boot_mean_hbr[trial_type].copy()


foo1[np.abs(foo) < t_crit] = 0 #np.nan # remove non-significant correlations
foo1 = foo1.reshape((n_channels,n_channels))
foo = foo.reshape((n_channels,n_channels))



node_angles = circular_layout(
    unique_parcels_list, node_order, start_pos=90, group_boundaries=[0, len(unique_parcels_list) / 2]
)

# make label_colors from the FreeSurfer parcellation
label_colors = np.zeros((len(unique_parcels_list), 4))
label_colors[:, 0] = 1.0
label_colors[:, 1] = 0.0
label_colors[:, 2] = 0.0
label_colors[:, 3] = 1.0
for ii in range(0, len(unique_parcels_list)):
    if unique_parcels_list[ii].endswith('LH'):
        label_colors[ii, 0] = 0.0
        label_colors[ii, 1] = 1.0
        label_colors[ii, 2] = 0.0
    elif unique_parcels_list[ii].endswith('RH'):
        label_colors[ii, 0] = 0.0
        label_colors[ii, 1] = 0.0
        label_colors[ii, 2] = 1.0


if flag_show_specific_parcels:
    for ii in range(0, len(parcels_to_show)):
        idx1 = [i for i, x in enumerate(unique_parcels_list) if x == parcels_to_show[ii]]
        idx2 = np.where(np.abs(foo1[:,idx1])>0)[0]
        idx2 = idx2[ np.where(idx2 != idx1[0])[0] ]# remove the diagonal
        if ii == 0:
            if flag_show_tstat:
                foo2 = foo[idx2,idx1]
            else:
                foo2 = foo1[idx2,idx1]
            indices1 = idx1 * np.ones((len(idx2))).astype(int)
            indices2 = idx2
        else:
            if flag_show_tstat:
                foo2 = np.concatenate((foo2, foo[idx2,idx1]), axis=0)
            else:
                foo2 = np.concatenate((foo2, foo1[idx2,idx1]), axis=0)
            indices1 = np.concatenate((indices1, idx1 * np.ones((len(idx2))).astype(int) ), axis=0)
            indices2 = np.concatenate((indices2, idx2), axis=0)
    indices = np.array([indices1, indices2])
else:
    if flag_show_tstat:
        foo2 = foo
    else:
        foo2 = foo1


# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig, ax = p.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
if flag_show_specific_parcels:
    plot_connectivity_circle(
        foo2,
        unique_parcels_list,
        indices=indices,
        n_lines=n_lines,
        node_angles=node_angles,
        node_colors=label_colors,
    #    title="All-to-All Connectivity left-Auditory " "Condition (PLI)",
        ax=ax,
        colormap=colormap,
        vmin=vminmax[0],
        vmax=vminmax[1],
    )
else:
    plot_connectivity_circle(
        foo2,
        unique_parcels_list,
        n_lines=n_lines,
        node_angles=node_angles,
        node_colors=label_colors,
    #    title="All-to-All Connectivity left-Auditory " "Condition (PLI)",
        ax=ax,
        colormap=colormap,
        vmin=vminmax[0],
        vmax=vminmax[1],
    )
fig.tight_layout()




# %%
# Unique parcel list with channel sensitivity for Sudan and Shannon
cfg_img_recon = {
    'probe_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/Laura_WMandRest/derivatives/fw',
    'head_model' : 'ICBM152',
    }

if 'Adot' not in locals():
    Adot, head = pfDAB_img.load_Adot( cfg_img_recon['probe_dir'], cfg_img_recon['head_model'])

# reduce parcels to 17 network parcels plus 'Background+Freesurfer...'
# get the unique 17 network parcels
unique_parcels = Adot.groupby('parcel').sum('vertex').parcel
unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'scalp')
unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')


parcel_list = []
for parcel in unique_parcels.values:
    parcel_list.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
unique_parcels_lev1 = np.unique(parcel_list)

parcel_list_lev2 = []
for parcel in unique_parcels.values:
    if parcel.split('_')[1].isdigit():
        parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
    else:
        parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[1] + '_' + parcel.split('_')[-1] )
unique_parcels_lev2 = np.unique(parcel_list_lev2)


Adot_parcels = Adot.isel(wavelength=0).groupby('parcel').sum('vertex')
Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'scalp')
Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')


Adot_parcels_lev1 = np.zeros( (Adot.shape[0], len(unique_parcels_lev1)) )
for ii in range( 0, len(unique_parcels_lev1) ):
    idx1 = [i for i, x in enumerate(parcel_list) if x == unique_parcels_lev1.tolist()[ii]]
    Adot_parcels_lev1[:,ii] = np.sum(Adot_parcels.isel(parcel=idx1).values, axis=1)

Adot_parcels_lev1_xr = xr.DataArray(
    Adot_parcels_lev1,
    dims=['channel','parcel_lev1'],
    coords={'channel': Adot.channel, 'parcel_lev1': unique_parcels_lev1}
)


Adot_parcels_lev2 = np.zeros( (Adot.shape[0], len(unique_parcels_lev2)) )
for ii in range( 0, len(unique_parcels_lev2) ):
    idx1 = [i for i, x in enumerate(parcel_list_lev2) if x == unique_parcels_lev2.tolist()[ii]]
    Adot_parcels_lev2[:,ii] = np.sum(Adot_parcels.isel(parcel=idx1).values,axis=1)

Adot_parcels_lev2_xr = xr.DataArray(
    Adot_parcels_lev2,
    dims=['channel','parcel_lev2'],
    coords={'channel': Adot.channel, 'parcel_lev2': unique_parcels_lev2}
)


# get index of element in unique_parcels_lev2
idx1 = [i for i, x in enumerate(unique_parcels_lev2) if x == 'SomMotA_RH']
# find index of max element in Adot_parcels_levs[:,idx1]
idx2 = np.argmax(Adot_parcels_lev2[:,idx1])

(idx2,idx1)
Adot.channel[idx2].values

# %%
# test scalp plot of pruned channels
importlib.reload(plots)


# Define discrete colors matching the criteria
# Create a discrete colormap and corresponding normalization
colors = ['red', (1,0.9,0.4), (0.3, 1, 0.3), 'cyan', 'blue', 'magenta']  # Change these colors if needed
bounds = [0.0, 0.19, 0.4, 0.65, 0.8, 0.95, 1]  # Boundaries for categories

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# plot
f,axs = p.subplots(1, 1, figsize=(8, 8))

cb_ticks_labels = [(0.1,'Saturated'), (0.3,'Poor SNR'), (0.52,'Good SNR'), (0.72,'SDS'), (0.87,'Low Signal'), (0.975,'SCI/PSP')]
ax1 = axs
plots.scalp_plot( 
        rec[0][0]["amp"],
        rec[0][0].geo3d,
        chs_pruned_subjs[0][0].values, 
        ax1,
        cmap=cmap,#'gist_rainbow',
        norm=norm,
        vmin=0,
        vmax=1,
        optode_labels=True,
        title="",
        optode_size=6,
        cb_ticks_labels = cb_ticks_labels
    )


# %%
# scalp plot the time series in channel space
importlib.reload(plots)
from cedalion.plots import scalp_plot_gif 

if 0:
    data_ts = od_ts.sel(wavelength=850)
    str_title = '850nm'
    scl = (-0.01,0.01)
else:
    data_ts = conc_ts.sel(chromo='HbO')
    str_title = 'HbO'
    scl = (-50,50)

geo3d = rec[0][0].geo3d
filename = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'image_recon', 'scalp_plot_ts')


scalp_plot_gif( 
    data_ts, 
    geo3d, 
    filename = filename, 
    time_range=(0,30,0.5)*units.s,
    scl=scl, 
    fps=6, 
    cmap='jet',
    optode_size=6, 
    optode_labels=True, 
    str_title=str_title
    )





# %%
# Scalp Plot correlation with specific channel seed

ch = ['S1D39','S7D39','S7D68','S29D68','S29D7','S29D6','S29D1','S56D1','S56D22','S56D18','S25D18','S25D6','S25D12']
idx = np.where( conc_ts.channel.values == ch[11] )[0]


f,axs = p.subplots(1,2,figsize=(16,8))

ax1 = axs[0]
plots.scalp_plot( 
    rec[0][0]["amp"],
    rec[0][0].geo3d,
    corr_hbo[:,idx].reshape(-1),
    ax1,
    cmap='jet',
    vmin=-1,
    vmax=1,
    optode_labels=True,
    title=f'HbO Correlation with {conc_ts.channel.values[idx]}',
    optode_size=6
)

ax1 = axs[1]
plots.scalp_plot( 
    rec[0][0]["amp"],
    rec[0][0].geo3d,
    corr_hbr[:,idx].reshape(-1),
    ax1,
    cmap='jet',
    vmin=-1,
    vmax=1,
    optode_labels=True,
    title=f'HbR Correlation with {conc_ts.channel.values[idx]}',
    optode_size=6
)

# %%
# gif of correlation with specific channel seed moving between channels

data_ts = xr.DataArray(
    np.zeros((corr_hbo.shape[0],len(ch))),
    dims=['channel','time'],
    coords={'channel': conc_ts.channel.values, 'time': np.arange(0,len(ch)) },
)
data_ts = data_ts.assign_coords(channel=('channel', conc_ts.channel.values))
data_ts = data_ts.assign_coords(source=('channel', conc_ts.source.values))
data_ts = data_ts.assign_coords(detector=('channel', conc_ts.detector.values))

for idx_ch, curr_ch in enumerate(ch):
    idx = np.where( conc_ts.channel.values == curr_ch )[0]
    data_ts.loc[dict(time=idx_ch)] = corr_hbo[:,idx].reshape(-1)

str_title = 'HbO'
scl = (-1,1)

geo3d = rec[0][0].geo3d
frame_range = (0,13,1)
filename = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'image_recon', 'scalp_plot_ts.gif')

v_spg.scalp_plot_gif( data_ts, geo3d, frame_range, filename, scl=scl, fps=1, optode_size=6, optode_labels=True, str_title=str_title )

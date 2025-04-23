# %% Imports
##############################################################################
#%matplotlib widget

import os
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils

import xarray as xr
import cedalion.plots as plots
from cedalion import units
import numpy as np

import gzip
import pickle
import json

import sys
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2/modules')
import module_image_recon as img_recon 
import module_spatial_basis_funs_ced as sbf 


# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')


#%%
# Have summed wt for each vertices across subjects


#%% Load in cfg pickle file
root_dir = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/"  # CHANGE
derivatives_subfolder = "Shannon"   # CHANGE
cfg_pkl_name = "cfg_params_BS_tddr_GLMfilt_unpruned.pkl"    # CHANGE   -- this is your config file name

#cfg_pkl_name = "cfg_params_STS_tddr_GLMfilt_unpruned.pkl" # STS
#cfg_pkl_name = "cfg_params_IWHD_imuGLM_tddr_GLMfilt_unpruned.pkl"  # IWHD

cfg_filepath = os.path.join(root_dir, 'derivatives', derivatives_subfolder, 'processed_data', cfg_pkl_name) 

# Open the file in binary read mode and load its contents
with open(cfg_filepath, 'rb') as file:
    cfg_params = pickle.load(file)

cfg_hrf = cfg_params["cfg_hrf"]
cfg_dataset = cfg_params["cfg_dataset"]
cfg_GLM = cfg_params["cfg_GLM"]
cfg_preprocess = cfg_params["cfg_preprocess"]
cfg_blockavg = cfg_params["cfg_blockavg"]
cfg_motion_correct = cfg_preprocess["cfg_motion_correct"]

subj_ids_new = [s for s in cfg_dataset['subj_ids'] if s not in cfg_dataset['subj_id_exclude']]


# %% Initial root directory and analysis parameters
##############################################################################

cfg_sb = {
    'mask_threshold': -2,
    'threshold_brain': 1*units.mm,      # threshold_brain / threshold_scalp: Defines spatial limits for brain vs. scalp contributions.
    'threshold_scalp': 5*units.mm,
    'sigma_brain': 1*units.mm,      # sigma_brain / sigma_scalp: Controls smoothing or spatial regularization strength.
    'sigma_scalp': 5*units.mm,
}


cfg_img_recon = {
    'probe_dir' : "/projectnb/nphfnirs/ns/Shannon/Data/probes/NN22_WHHD/12NN/",  #/fw ?
    'head_model' : 'ICBM152',
    'img_recon_on_group' : False,
    't_win' : (10, 20), 
    'DIRECT' : False,  # If true, does direct method, False = does indirect
    'flag_Cmeas' : True,
    'BRAIN_ONLY' : False,
    'SB' : True,    # spatial basis
    'alpha_meas' : 1e-2,  # measurement regularization
    'alpha_spatial' : 1e-2,    #  spatial reg , small pushes deeper into the brain   -- use 1e-2 WITH SB, 1e-3 WITHOUT SB
    'spectrum' : 'prahl',
    'cfg_sb' : cfg_sb,
    'flag_save_img_results' : True
    }

mse_min_thresh = 1e-3 

save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'], 'processed_data')

# !!! SAVE image recon configs in another json 
# !!! ADD flag for if doing image recon on ts or hrf mag ?


#%% Load Saved data

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

p_save_str =  p_save_str + '_OD' 

filname =  'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl.gz'

print("Loading saved data")

filepath_bl = os.path.join(save_path , filname) 
if os.path.exists(filepath_bl):
    with gzip.open(filepath_bl, 'rb') as f:
        groupavg_results = pickle.load(f)
    blockaverage_mean = groupavg_results['blockaverage']
    blockaverage_stderr = groupavg_results['blockaverage_stderr']
    blockaverage_subj = groupavg_results['blockaverage_subj']
    blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    geo2d = groupavg_results['geo2d']
    geo3d = groupavg_results['geo3d']
    print(f" {filname} loaded successfully!")

else:
    print(f"Error: File '{filepath_bl}' not found!")
        
blockaverage_all = blockaverage_mean.copy()


#%% load head model 
#probe_dir = "/projectnb/nphfnirs/ns/Shannon/Data/probes/NN22_WHHD/12NN/" 
import importlib
importlib.reload(img_recon)

head, PARCEL_DIR = img_recon.load_head_model(cfg_img_recon['head_model'], with_parcels=False)
Adot, meas_list, geo3d, amp = img_recon.load_probe(cfg_img_recon['probe_dir'], snirf_name='fullhead_56x144_NN22_System1.snirf')


ec = cedalion.nirs.get_extinction_coefficients('prahl', Adot.wavelength)
einv = cedalion.xrutils.pinv(ec)

#%% run image recon

"""
do the image reconstruction of each subject independently 
- this is the unweighted subject block average magnitude 
- then reconstruct their individual MSE
- then get the weighted average in image space 
- get the total standard error using between + within subject MSE 
"""
threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)


#ind_subj_blockavg = all_results['ind_subj_blockavg_unweighted']
ind_subj_blockavg = groupavg_results['blockaverage_subj']  
#ind_subj_mse = all_results['ind_subj_mse']
ind_subj_mse = groupavg_results['blockaverage_mse_subj']

F = None
D = None
G = None

all_trial_X_hrf_mag = None

for trial_type in ind_subj_blockavg.trial_type:
    
    print(f'Getting images for trial type = {trial_type.values}')
    all_subj_X_hrf_mag = None
    
    for subj in ind_subj_blockavg.subj:
        print(f'Calculating subject = {subj.values}')

        od_hrf = ind_subj_blockavg.sel(subj=subj, trial_type=trial_type) 
        # od_hrf = od_hrf.stack(measurement=('channel', 'wavelength')).sortby('wavelength')

        od_mse = ind_subj_mse.sel(subj=subj, trial_type=trial_type).drop_vars(['subj', 'trial_type'])
        
        od_hrf_mag = od_hrf.sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
        od_mse_mag = od_mse.sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
        
        C_meas = od_mse_mag.pint.dequantify()
        C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        C_meas = xr.where(C_meas < mse_min_thresh, mse_min_thresh, C_meas)

            

        X_hrf_mag, W, D, F, G = img_recon.do_image_recon(od_hrf_mag, head = head, Adot = Adot, C_meas_flag = cfg_img_recon['flag_Cmeas'], C_meas = C_meas, 
                                                    wavelength = [760,850], BRAIN_ONLY = cfg_img_recon['BRAIN_ONLY'], DIRECT = cfg_img_recon['DIRECT'], SB = cfg_img_recon['SB'], 
                                                    cfg_sbf = cfg_img_recon['cfg_sb'], alpha_spatial = cfg_img_recon['alpha_spatial'], alpha_meas = cfg_img_recon['alpha_meas'],
                                                    F = F, D = D, G = G)

        

        X_mse = img_recon.get_image_noise(C_meas, X_hrf_mag, W, DIRECT = cfg_img_recon['DIRECT'], SB= cfg_img_recon['SB'], G=G)
        

        # weighted average -- same as chan space - but now is vertex space
        if all_subj_X_hrf_mag is None:
            
            all_subj_X_hrf_mag = X_hrf_mag
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(subj=subj)
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(trial_type=trial_type)

            all_subj_X_mse = X_mse
            all_subj_X_mse = all_subj_X_mse.assign_coords(subj=subj)
            all_subj_X_mse = all_subj_X_mse.assign_coords(trial_type=trial_type)

            X_hrf_mag_weighted = X_hrf_mag / X_mse
            X_mse_inv_weighted = 1 / X_mse   # X_mse = mse for 1 subject across all vertices , inverse is wt
            
        else:

            X_hrf_mag_tmp = X_hrf_mag.assign_coords(subj=subj)
            X_hrf_mag_tmp = X_hrf_mag_tmp.assign_coords(trial_type=trial_type)

            X_mse_tmp = X_mse.assign_coords(subj=subj)
            X_mse_tmp = X_mse_tmp.assign_coords(trial_type=trial_type)

            all_subj_X_hrf_mag = xr.concat([all_subj_X_hrf_mag, X_hrf_mag_tmp], dim='subj')
            all_subj_X_mse = xr.concat([all_subj_X_mse, X_mse_tmp], dim='subj')

            X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
            X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse       # summing weight over all subjects -- viz X_mse_inv_weighted will tell us which regions of brain we are most conf in
        # END OF SUBJECT LOOP

    # get the average
    X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
    X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
    
    X_mse_mean_within_subject = 1 / X_mse_inv_weighted
    X_mse_mean_within_subject = X_mse_mean_within_subject.assign_coords({'trial_type': trial_type})
    
    X_mse_subj_tmp = all_subj_X_mse # PLOT THIS
    
    # temp = all_subj_X_mse.copy()
    # temp[: ~M] = np.nan
    # temp = np.log10(temp.sel(vertex=all_subj_X_mse.is_brain.values).stack(val=('vertex', 'chromo', 'subj')))
    # temp[np.isneginf(temp)] = np.nan
    
    # plt.hist(temp, bins=100)
    # plt.axvline(np.log10(mse_min_thresh), color='k')
    
    # X_mse_subj_tmp = xr.where(X_mse_subj_tmp < mse_min_thresh, mse_min_thresh, X_mse_subj_tmp)
    X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean)**2 / X_mse_subj_tmp # X_mse_subj_tmp is weights for each sub
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects * X_mse_mean_within_subject # / (all_subj_X_mse**-1).mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.pint.dequantify()
    
    X_stderr_weighted = np.sqrt( X_mse_mean_within_subject + X_mse_weighted_between_subjects )
    
    X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted
    
    if all_trial_X_hrf_mag is None:
        
        all_trial_X_hrf_mag = X_hrf_mag_mean
        all_trial_X_hrf_mag_weighted = X_hrf_mag_mean_weighted
        all_trial_X_stderr = X_stderr_weighted
        all_trial_X_tstat = X_tstat
        all_trial_X_mse_between = X_mse_weighted_between_subjects
        all_trial_X_mse_within = X_mse_mean_within_subject
    else:

        all_trial_X_hrf_mag = xr.concat([all_trial_X_hrf_mag, X_hrf_mag_mean], dim='trial_type')
        all_trial_X_hrf_mag_weighted = xr.concat([all_trial_X_hrf_mag_weighted, X_hrf_mag_mean_weighted], dim='trial_type')
        all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim='trial_type')
        all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')
        all_trial_X_mse_between = xr.concat([all_trial_X_mse_between, X_mse_weighted_between_subjects], dim='trial_type')
        all_trial_X_mse_within = xr.concat([all_trial_X_mse_within, X_mse_mean_within_subject], dim='trial_type')

# END OF TRIAL TYPE LOOP
results = {'X_hrf_mag': all_trial_X_hrf_mag,
           'X_hrf_mag_weighted': all_trial_X_hrf_mag_weighted,
           'X_std_err': all_trial_X_stderr,
           'X_tstat': all_trial_X_tstat,
           'X_mse_between': all_trial_X_mse_between,
           'X_mse_within': all_trial_X_mse_within
           }

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
    
der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'], 'processed_data', 'image_recon')
if not os.path.exists(der_dir):
    os.makedirs(der_dir)

# if cfg_img_recon['SB']:
#     filepath = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'], 'processed_data', 'image_recon', f'Xs_{cfg_dataset["file_ids"][0].split("_")[0]}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}.pkl.gz')
#     print(f'   Saving to Xs_{cfg_dataset["file_ids"][0].split("_")[0]}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}_sigma_brain_{cfg_sb["sigma_brain"]}_sigma_scalp_{cfg_sb["sigma_scalp"]}.pkl.gz')
#     file = gzip.GzipFile(filepath, 'wb')
#     file.write(pickle.dumps(results))
#     file.close()     
# else:
    
# Save results
filepath = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'processed_data', 'image_recon', f'Xs_{cfg_dataset["file_ids"][0].split("_")[0]}_{p_save_str}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}.pkl.gz')
print(f'   Saving to Xs_{cfg_dataset["file_ids"][0].split("_")[0]}_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps(results))
file.close()     


# for single subject image recon ---- X_mse would be what to use for single sub
    # X_mse is standard error^2 
    # X_tstat = magnitude of sqrt root divided by X_mse



#%% Save image configs in pickle and json file

fil_name = f'_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}'
# SAVE cfg params to json file
dict_cfg_save = {"cfg_sb": cfg_sb, "cfg_img_recon" : cfg_img_recon}

cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + fil_name
save_json_path = os.path.join(save_path, 'image_recon', cfg_save_str + '.json')
save_pickle_path = os.path.join(save_path, 'image_recon', cfg_save_str + '.pkl')

 
# Save configs as json to view outside of python
with open(os.path.join(save_json_path), "w", encoding="utf-8") as f:
    json.dump(dict_cfg_save, f, indent=4, default = str)  # Save as JSON with indentation

# Save configs as Pickle for Python usage (preserving complex objects like Pint quantities)
with open(save_pickle_path, "wb") as f:
    pickle.dump(dict_cfg_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    
print(f'  Saving config parameters to {cfg_save_str}')
    

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
    
flag_condition_list = cfg_hrf['stim_lst']

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
                    save_dir_tmp= os.path.join(cfg_dataset["root_dir"], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'image_recon', img_folder)
                    if not os.path.exists(save_dir_tmp):
                        os.makedirs(save_dir_tmp)
                    file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
                    p0.screenshot( os.path.join(save_dir_tmp, file_name) )
                    p0.close()
                else:
                    p0.show()
                    


# 
#%%
# tasknm = cfg_dataset["file_ids"][0].split('_')[0]

# filepath = os.path.join(save_path, f'Xs_{tasknm}_direct_alltrial_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
# print(f'   Saving to Xs_{tasknm}_direct_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
# file = gzip.GzipFile(filepath, 'wb')
# file.write(pickle.dumps(results_img_s))
# file.close()   


#%% Load image recon results

# filname =  'Xs_STS_direct_alltrial_cov_alpha_spatial_1e-01_alpha_meas_1e+00.pkl.gz'
# filepath_bl = os.path.join(save_path , filname)

# if os.path.exists(filepath_bl):
#     with gzip.open(filepath_bl, 'rb') as f:
#         results_img_s = pickle.load(f)
#     all_trial_X_hrf_mag = results_img_s['X_hrf_mag_all_trial']
#     all_trial_X_hrf_mag_weighted = results_img_s['X_hrf_mag_weighted_all_trial']
#     all_trial_X_tstat = results_img_s['X_tstat_all_trial']
#     all_trial_X_stderr = results_img_s['X_std_err_all_trial']
    
#     print("Image results file loaded successfully!")


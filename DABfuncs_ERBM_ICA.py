import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.xrutils as xrutils
import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
#import cedalion.vis.time_series as vTimeSeries
from cedalion import units
import numpy as np
import pandas as pd

import json

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from sklearn.decomposition import PCA
from cedalion.sigdecomp.ERBM import ERBM
from scipy import stats




def ERBM_ICA( TS, stim, p_ica, pca_var_thresh, ica_tstat_thresh, ica_spatial_mask_thresh, trange_hrf, trange_hrf_stat, flag_use_saved_ica, flag_use_tddr, rootDir_data, filenm ):

    ts_mean = TS.mean('time')
    ts_std = TS.std('time')
    ts_zscore = stats.zscore(TS.values, axis=0)
    
    # get indices of mean_ts_zscore with NaN
    mean_ts_zscore = ts_zscore.mean(axis=0)
    idx_nan = np.where(np.isnan(mean_ts_zscore))[0]
    idx_not_nan = np.where(~np.isnan(mean_ts_zscore))[0]

    # remove columns with NaN
    ts_zscore = np.delete(ts_zscore, idx_nan, axis=1)

    #% run ICA algorithm
    # PCA on segment
    pca = PCA()
    S_pca = pca.fit_transform(ts_zscore)
    explained = pca.explained_variance_ratio_
    
    # Calculate cumulative explained variance
    cumulative_explained = np.cumsum(explained) #*100
    
    # Find the number of components required to meet the variance threshold
    num_components = np.argmax(cumulative_explained >= pca_var_thresh) + 1
    print(f'   number of PCA components kept: {num_components}')
    
    # Select the subset of principal components that meet the variance threshold
    S_pca_thresh = S_pca[:, :num_components]
    W_pca = pca.components_[:num_components, :]
    
    #% ICA-ERBM on PCs
    import time
    from datetime import datetime
    if not flag_use_saved_ica:
        start_time = time.time()
        print(f'start time = {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}')
        W_ica = ERBM(S_pca_thresh.T, p_ica)
        end_time = time.time()
        execution_time = end_time - start_time
        print("ERBM ICA execution time:", execution_time, "seconds")
    else:
        file_path = os.path.join(rootDir_data, 'derivatives', 'ica', filenm )
        if not flag_use_tddr:
            foo = np.load(file_path + '_Wica_conc.npz' )
        else:
            foo = np.load(file_path + '_Wica_conc_tddr.npz' )
        W_ica = foo['W_ica']
        
    S_ica = W_ica @ S_pca_thresh.T

    #
    # block average the ICA components
    #
    
    # create xarray
    S_ica_xr = xr.DataArray(
        S_ica.reshape(S_ica.shape[0], 1, S_ica.shape[1]),
        dims=["channel", "subject", "time"],
#        coords={"channel": np.arange(1, S_ica.shape[0]+1), "subject": [1], "time": rec[subj_idx][file_idx]["conc"].time.values * units.s}
        coords={"channel": np.arange(1, S_ica.shape[0]+1), "subject": [1], "time": TS.time.values * units.s}
    )
    S_ica_xr = S_ica_xr.assign_coords(samples=("time", np.arange(S_ica_xr.shape[2])))
    S_ica_xr = S_ica_xr.transpose("subject", "channel", "time")
    S_ica_xr.time.attrs['units'] = 'second'

    # get the blocks
    stim = stim.copy()
    S_ica_xr_epochs = S_ica_xr.cd.to_epochs(
        stim,  # stimulus dataframe
        ["ST"],  # select events    
        before=trange_hrf[0],  # seconds before stimulus 
        after=trange_hrf[1]  # seconds after stimulus
    )

    # baseline subtract
    S_ica_xr_epochs = S_ica_xr_epochs - S_ica_xr_epochs.sel(reltime=slice(-trange_hrf[0], 0)).mean("reltime")

    # get the mean and std
    blockaverage_mean = S_ica_xr_epochs.groupby("trial_type").mean("epoch")
    blockaverage_std = S_ica_xr_epochs.groupby("trial_type").std("epoch") 

    # t-stat
    t_stat = blockaverage_mean / (blockaverage_std / np.sqrt(S_ica_xr_epochs.sizes["epoch"]))

    # get the mean t-stat from reltime 5 to 20
    t_stat_mean = t_stat.sel(reltime=slice(trange_hrf_stat[0], trange_hrf_stat[1])).mean("reltime")

    # get component indices that are significant
    idx_sig = np.where(np.abs(t_stat_mean[0,0,:]) > ica_tstat_thresh)[0]

    #
    # get the components with broad spatial distribution in the same direction
    #

    # set new_ts to NaN with the same shape as ts_zscore
    T_ica_to_ch = np.full((W_ica.shape[0],ts_std.shape[0]), np.nan)
    # project back to channel space
    T_ica_to_ch[:,idx_not_nan] = np.linalg.pinv(W_ica).T @ W_pca

    # get the norm of each component ignoring the NaNs
    norm = np.sqrt( np.nansum( T_ica_to_ch * T_ica_to_ch, axis=1 ) )

    foo = T_ica_to_ch / norm[:, np.newaxis]
    metric = np.nanmean(foo, axis=1)
    metric1 = np.nanmean(foo[:,:567], axis=1)
    metric2 = np.nanmean(foo[:,567:-1], axis=1)

    idx_remove = np.where( abs(metric) > (ica_spatial_mask_thresh * np.std(metric))  )[0]

    # keep indices from idx_sig that are not in idx_remove
    idx_sig_minus_remove = np.setdiff1d(idx_sig, idx_remove)

    # plot the metric
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].plot(abs(metric))
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].plot(idx_sig, abs(metric[idx_sig]), 'go', markerfacecolor='none')
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].plot(idx_remove, abs(metric[idx_remove]), 'rx')
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].set_title(f"s{idx_file}, #C={len(idx_sig_minus_remove)}")

    print(f'   number of ICA components kept: {len(idx_sig_minus_remove)}')

    #
    # project ICA back to channel space
    #

    # get the indices of the components to project back
    idx = idx_sig_minus_remove
                        
    # S_ica_idx contains the components you want to project back, otherwise zeros 
    S_ica_idx = np.zeros_like(S_ica)
    S_ica_idx[idx, :] = S_ica[idx, :]
                                    
    # project back to PC space
    S_pca_idx = np.linalg.pinv(W_ica) @ S_ica_idx

    # set new_ts to NaN with the same shape as ts_zscore
    new_ts = np.full((ts_zscore.shape[0],ts_std.shape[0]), np.nan)

    # project back to channel space
    new_ts[:,idx_not_nan] = S_pca_idx.T @ W_pca
    #new_ts = S_pca_idx.T @ W_pca

    # scale the columns of new_ts by ts_std
    ts_std_values = ts_std.values
    new_ts[:,idx_not_nan] = new_ts[:,idx_not_nan] @ np.diag(ts_std_values[idx_not_nan]) + ts_mean[idx_not_nan].values

    new_xr = xr.zeros_like(TS)
    new_xr.values = new_ts * units.micromolar
    new_xr = new_xr.unstack("measurement")

    detector_coord = new_xr["detector"].data[:, 0]
    new_xr = new_xr.assign_coords(detector=("channel", detector_coord))
    source_coord = new_xr["source"].data[:, 0]
    new_xr = new_xr.assign_coords(source=("channel", source_coord))

    new_xr = new_xr.transpose("chromo", "channel", "time")
    new_xr.time.attrs['units'] = 'second'

    #
    # remove the GMS
    #
    conc_ica = new_xr.copy()
    conc_ica_values = conc_ica.values

    if 1: # remove global mean
        y_mean = np.nanmean(conc_ica_values, axis=1)
    else:
        # get first principal component of conc_ica_values
        pca = PCA(n_components=1)
        y_mean = np.zeros((conc_ica_values.shape[2], conc_ica_values.shape[0]))
        for i in range(conc_ica_values.shape[0]):
            foo = conc_ica_values[i,:,:].T
            #list of columns with no NaN values
            foo = foo[:,~np.isnan(foo).any(axis=0)]
            # Fit PCA on the data and transform it to get the first principal component
            y_mean[:,i] = np.squeeze(pca.fit_transform(foo))
        y_mean = y_mean.T
        
    numerator = np.nansum(conc_ica_values * y_mean[:, np.newaxis], axis=2)
    denominator = np.nansum(y_mean * y_mean, axis=1) 
    denominator = denominator[:, np.newaxis] @ np.ones((1,numerator.shape[1]))

    scl = numerator/denominator                    
    scl = scl[:,:,np.newaxis] @ np.ones((1,conc_ica_values.shape[2]))                 

    conc_ica_values = conc_ica_values - scl*y_mean[:, np.newaxis]

    conc_ica_gms = conc_ica.copy()
    conc_ica_gms.values = conc_ica_values * units.micromolar

    return conc_ica, conc_ica_gms 




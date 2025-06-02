import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.xrutils as xrutils
import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform




def getCorrMatrix( conc_hbo = None, flag_GMS = "all" ):

        # global mean subtract conc_hbo
    if flag_GMS == "all": # do GMS with all channels or SS channels
        # select all channels
        gms = conc_hbo.mean(dim='channel')

        # fit GMS to the channel data
        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator

        conc_hbo_gms = conc_hbo - scl*gms

    elif flag_GMS == "ss": # do GMS with SS channels
        # select SS channels only
        gms = conc_hbo.sel(channel=conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')
        # FIX ME. Quality.sd_dist should be used to determine which channels are SS and LS

        # fit GMS to the channel data
        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator

        conc_hbo_gms = conc_hbo - scl*gms

    elif flag_GMS == "ss_ls": # do GMS with SS channels and then LS channels
        # SS channels
        gms = conc_hbo.sel(channel=conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')

        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator
        conc_hbo_gms = conc_hbo - scl*gms

        # LS channels
        gms = conc_hbo_gms.sel(channel=~conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')

        numerator = (conc_hbo_gms * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator
        conc_hbo_gms = conc_hbo_gms - scl*gms

    else:
        conc_hbo_gms = conc_hbo
        
    # calculate the correlation matrix of conc_hbo
    conc_hbo_array = conc_hbo_gms.values

    if 1: # correlation matrix
        corr_matrix = np.corrcoef(conc_hbo_array, rowvar=True)
        corr_max = 1

    else: # partial correlation matrix
        regularization = 1e2
        # Step 1: Calculate the covariance matrix
        cov_matrix = np.cov(conc_hbo_array, rowvar=True)    
        # Step 2: Regularize by adding a small value to the diagonal
        cov_matrix += np.eye(cov_matrix.shape[0]) * regularization    
        # Step 3: Invert the covariance matrix to get the precision matrix
        precision_matrix = np.linalg.inv(cov_matrix)
        # Step 4: Convert the precision matrix to the partial correlation matrix
        partial_corr = precision_matrix / np.sqrt(np.outer(np.diag(precision_matrix), np.diag(precision_matrix)))
        # Step 5: Ensure the diagonal is all zeros
        np.fill_diagonal(partial_corr, 0)

        corr_matrix = partial_corr
#        corr_max = np.max(np.abs(corr_matrix))

        

    # Ensure corr_matrix_xr has two dimensions, both of which are channel
    corr_matrix_xr = xr.DataArray(
        corr_matrix,
    #    cov_matrix,
        dims=["channel", "channel"],
        coords={"channel": conc_hbo.channel.values, "channel": conc_hbo.channel.values}
    )

    return corr_matrix_xr, conc_hbo_gms


def calc_dFC( conc_hbo_array_clusters = None, t = None, window_size_s = 20 ):

    fs = 1 / np.mean(np.diff(t))
    window_size = window_size_s * fs
    window_size = int(window_size.round())

    max_cluster_label = conc_hbo_array_clusters.shape[0]

    # initialize the correlation matrix
    corr_time_clusters = np.zeros([max_cluster_label, max_cluster_label, conc_hbo_array_clusters.shape[1] - window_size])

    # get the tcorr vector
    # tcorr = np.zeros(corr_time_clusters.shape[2])
    # for i in range(0, corr_time_clusters.shape[2], 1):
    #     tcorr[i] = t[i + window_size // 2]
    # Assuming tcorr and t are NumPy arrays and window_size is defined
    tcorr = t[window_size // 2 : corr_time_clusters.shape[2] + window_size // 2]

    # For loop over c1 
    for c1 in range(0, max_cluster_label):
        for c2 in range(c1, max_cluster_label):

            if c1 == c2:
                corr_time_clusters[c1,c2,:] = conc_hbo_array_clusters[c1,window_size//2:-window_size//2]
            else:
                # calculate the correlation between the two clusters over a sliding window
                y1 = conc_hbo_array_clusters[c1,:]
                y2 = conc_hbo_array_clusters[c2,:]

                # calculate the correlation between the two channels over a sliding window
                for i in range(0, y1.shape[0] - window_size, 1):
                    corr_time_clusters[c1,c2,i] = np.corrcoef(y1[i:i + window_size], y2[i:i + window_size])[0, 1]

    return corr_time_clusters, tcorr


def block_average_clusters( corr_time_clusters, tcorr, stim, events_str, t_before = 2, t_after = 20 ):
    
    max_cluster_label = corr_time_clusters.shape[0]

    if len(events_str) > 2:
        print("Only two events are allowed, but you can fix the code")
        return
    
    duration = np.zeros(2)
    stim_trial_type = stim.loc[stim.trial_type == events_str[0]]
    duration[0] = stim_trial_type.duration.mean()
    stim_trial_type = stim.loc[stim.trial_type == events_str[1]]
    duration[1] = stim_trial_type.duration.mean()

    if duration[0] != duration[1]:
        print("The duration of the two events must be the same, but you can fix the code")
        return

    # put corr_time_clusters into an xarray
    corr_time_clusters_xr = xr.DataArray(
        corr_time_clusters,
        dims=["cluster1", "cluster2", "time"],
        coords={"cluster1": np.arange(1, max_cluster_label+1), "cluster2": np.arange(1, max_cluster_label+1), "time": tcorr}
    )
    corr_time_clusters_xr = corr_time_clusters_xr.assign_coords(samples=("time",np.arange(corr_time_clusters_xr.shape[2])))

    # get the blocks
    corr_time_clusters_xr_epochs = corr_time_clusters_xr.cd.to_epochs(
        stim,  # stimulus dataframe
        events_str,  # select events
        t_before,  # seconds before stimulus
        t_after,  # seconds after stimulus
    )

    # block average
    blockaverage_clusters = corr_time_clusters_xr_epochs.groupby("trial_type").mean("epoch")

    # get t-stat
    # FIX ME - this assumes stim duration is the same for all trials
    foo = corr_time_clusters_xr_epochs.sel(reltime=(corr_time_clusters_xr_epochs.reltime > 0) & (corr_time_clusters_xr_epochs.reltime < stim.duration.mean())).mean("reltime")
    blockaverage_clusters_mean = foo.groupby("trial_type").mean("epoch")
    blockaverage_clusters_std = foo.groupby("trial_type").std("epoch")
    blockaverage_clusters_sem = foo.groupby("trial_type").std("epoch") / np.sqrt(foo.epoch.size)
    # FIX ME - CHECK that this is dividing by the correct number of epochs 

    # FIX ME - this assumes only two trials types
    # calculate the t-statistic between the two trial_types
    blockaverage_clusters_tstat = ( blockaverage_clusters_mean[0,:,:] - blockaverage_clusters_mean[1,:,:] ) / np.sqrt( blockaverage_clusters_sem[0,:,:]**2 + blockaverage_clusters_sem[1,:,:]**2 )

    return blockaverage_clusters, blockaverage_clusters_tstat, blockaverage_clusters_mean, blockaverage_clusters_std, blockaverage_clusters_sem



def corr_cluster( corr_matrix_xr, cluster_threshold ):
    
    # calculate the distance matrix
    #dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = 1 - corr_matrix_xr.values

    # Fill NaN values with a specific value (e.g., 0)
    dist_matrix = np.nan_to_num(dist_matrix, nan=2)

    # Ensure the distance matrix is symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Ensure the diagonal of the distance matrix is zero
    np.fill_diagonal(dist_matrix, 0)

    # calculate the linkage matrix
    linkage_matrix = linkage(squareform(dist_matrix), method="average")

    return linkage_matrix



def preprocess_dataset( rec, chs_pruned_subjs, cfg_dataset, cfg_blockavg, unique_trial_types, 
                       Adot_parcels_lev1_xr, Adot_parcels_lev2_xr,
                       flag_do_bp_filter_on_od = True, flag_do_AR_filter_on_conc = True, flag_do_gms_chromo = True, flag_channels_to_parcels = False, flag_parcels_use_lev1 = False ):

    n_subjects = len(cfg_dataset['subj_ids'])
    n_files = len(cfg_dataset['file_ids'])

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

            # Do the AR filtering
            if flag_do_AR_filter_on_conc:
                ar_order = 7
                conc_ts_tmp = conc_ts.isel(time=slice(ar_order, None)).copy()  # remove the first ar_order points
                for idx_chromo, chromo in enumerate(conc_ts.chromo):
                    for idx_ch, ch in enumerate(conc_ts.channel):
                        y = conc_ts.sel(chromo=chromo, channel=ch).values

                        # Fit AR model if y has no NaNs or Infs
                        if np.all(np.isfinite(y)):
                            model = sm.tsa.AutoReg(y, lags=ar_order, old_names=False)
                            result = model.fit()
                            residuals = result.resid * units.micromolar  # convert back to micromolar

                            # Store the residuals back into the time series
                            conc_ts_tmp.loc[dict(chromo=chromo, channel=ch)] = xr.DataArray(residuals, dims='time', coords={'time': conc_ts_tmp.time})
                conc_ts = conc_ts_tmp.copy()

            # Global mean subtraction for each chromo
            if flag_do_gms_chromo:
                # do a weighted mean by variance
                conc_var = conc_ts.var('time')

                # correct for bad data
                amp = rec[idx_subj][idx_file]['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
                idx_amp = np.where(amp < cfg_blockavg['cfg_mse_od']['mse_amp_thresh'])[0]
                conc_var.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
                conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_amp).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

                idx_sat = np.where(chs_pruned_subjs[idx_subj][idx_file] == 0.0)[0] 
                conc_var.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data']
                conc_ts.loc[dict(channel=conc_ts.isel(channel=idx_sat).channel.data)] = cfg_blockavg['cfg_mse_conc']['blockaverage_val']

                # FIXME: deal with rare instances when conc_var is 0

                gms = (conc_ts / conc_var).mean('channel') / (1/conc_var).mean('channel')
                numerator = (conc_ts * gms).sum(dim="time")
                denominator = (gms * gms).sum(dim="time")
                scl = numerator / denominator
                conc_ts = conc_ts - scl*gms

            # project channels to parcels
            if flag_channels_to_parcels: # project channels to parcel_lev2 by weighted average over channels
                w = 1 / conc_var
                # get the normalized weighted averaging kernel
                if flag_parcels_use_lev1:
                    Adot_parcels_weighted_xr = w * Adot_parcels_lev1_xr
                else:
                    Adot_parcels_weighted_xr = w * Adot_parcels_lev2_xr
                Adot_parcels_weighted_xr = Adot_parcels_weighted_xr / Adot_parcels_weighted_xr.sum(dim='channel')
                # do the inner product over channel between conc_ts and Adot_parcels_weighted_xr
                foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T @ conc_ts.sel(chromo='HbO')
                foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T @ conc_ts.sel(chromo='HbR')
                conc_ts = xr.concat([foo_hbo, foo_hbr], dim='chromo')
                # do the same with conc_var
                foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T**2 @ conc_var.sel(chromo='HbO')
                foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T**2 @ conc_var.sel(chromo='HbR')
                conc_var = xr.concat([foo_hbo, foo_hbr], dim='chromo')

                n_ch_or_parcel = conc_ts.parcel.size
            else:
                n_ch_or_parcel = conc_ts.channel.size

            # loop over trial types
            for idx_trial_type, trial_type in enumerate(unique_trial_types):
                if trial_type != 'full_ts':
                    idx = np.where(rec[idx_subj][idx_file].stim.trial_type==trial_type)[0]
                    t_indices_tmp = np.array([])
                    dt = np.median(np.diff(conc_ts.time)) 
                    for ii in idx:
                        t_indices_tmp = np.concatenate( (t_indices_tmp, np.where( 
                                            (conc_ts.time >  rec[idx_subj][idx_file].stim.onset[ii]) &
                                            (conc_ts.time <= (rec[idx_subj][idx_file].stim.onset[ii] + rec[idx_subj][idx_file].stim.duration[ii] )) 
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
                    # FIXME remove this next 4 lines once I am done testing
                    # corr_hbo_files[trial_type] = np.zeros((len(cfg_dataset['file_ids']), len(conc_ts.parcel)**2))
                    # corr_hbr_files[trial_type] = np.zeros((len(cfg_dataset['file_ids']), len(conc_ts.parcel)**2))
                    # corr_hbo_files[trial_type][idx_file, :] = corr_hbo.reshape(-1)
                    # corr_hbr_files[trial_type][idx_file, :] = corr_hbr.reshape(-1)
                else:
                    conc_ts_files[trial_type] = xr.concat([conc_ts_files[trial_type], foo_ts], dim='time', coords='minimal', compat='override') # ensure no reordering since times overlap
                    conc_var_files[trial_type] = xr.concat([conc_var_files[trial_type], conc_var], dim='file')
                    # remove these next two lines once I am done testing
                    # corr_hbo_files[trial_type][idx_file, :] = corr_hbo.reshape(-1)
                    # corr_hbr_files[trial_type][idx_file, :] = corr_hbr.reshape(-1)
                
                da_hbo = xr.DataArray(
                        corr_hbo.reshape(1,1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                        dims=["subj","file", "trial_type", "chromo", "correlation_A", "correlation_B"],
                        coords={"chromo": ["HbO"], "trial_type": [trial_type], "file": [idx_file], "subj": [curr_subj]}
                    )
                da_hbr = xr.DataArray(
                        corr_hbr.reshape(1,1,1,1,n_ch_or_parcel,n_ch_or_parcel),
                        dims=["subj","file", "trial_type", "chromo", "correlation_A", "correlation_B"],
                        coords={"chromo": ["HbR"], "trial_type": [trial_type], "file": [idx_file], "subj": [curr_subj]}
                    )
                if idx_trial_type == 0:
                    corr_trial_type = xr.concat([da_hbo, da_hbr], dim="chromo")
                else:
                    corr_trial_type_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                    corr_trial_type = xr.concat([corr_trial_type, corr_trial_type_tmp], dim="trial_type")
                
            # end of trial type loop

            if idx_file == 0:
                corr_files = corr_trial_type.copy()
            else:
                corr_files = xr.concat([corr_files, corr_trial_type], dim="file")

        # end of file loop

        # store the time series for each subject for each trial_type
        for idx_trial_type, trial_type in enumerate(unique_trial_types):
            conc_ts_subjs[idx_subj][trial_type] = conc_ts_files[trial_type].copy()
            conc_var_subjs[idx_subj][trial_type] = conc_var_files[trial_type].mean('file') 

        if idx_subj == 0:
            corr_subj_files = corr_files.copy()
        else:
            corr_subj_files = xr.concat([corr_subj_files, corr_files], dim="subj")

            # I am now doing this above. This can be deleted.
            # if flag_channels_to_parcels: # project channels to parcel_lev2 by weighted average over channels
            #     w = 1 / conc_var_subjs[idx_subj][trial_type]
            #     # get the normalized weighted averaging kernel
            #     if flag_parcels_use_lev1:
            #         Adot_parcels_weighted_xr = w * Adot_parcels_lev1_xr
            #     else:
            #         Adot_parcels_weighted_xr = w * Adot_parcels_lev2_xr
            #     Adot_parcels_weighted_xr = Adot_parcels_weighted_xr / Adot_parcels_weighted_xr.sum(dim='channel')
            #     # do the inner product over channel between conc_ts_subjs[idx_subj][trial_type] and Adot_parcels_lev2_weighted_xr
            #     foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T @ conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO')
            #     foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T @ conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR')
            #     conc_ts_subjs[idx_subj][trial_type] = xr.concat([foo_hbo, foo_hbr], dim='chromo')
            #     # do the same with conc_var_subjs[idx_subj][trial_type]
            #     foo_hbo = Adot_parcels_weighted_xr.sel(chromo='HbO').T**2 @ conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO')
            #     foo_hbr = Adot_parcels_weighted_xr.sel(chromo='HbR').T**2 @ conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR')
            #     conc_var_subjs[idx_subj][trial_type] = xr.concat([foo_hbo, foo_hbr], dim='chromo')


        # get the repeatability
        for idx_trial_type, trial_type in enumerate(unique_trial_types):
            foo_hbo = np.corrcoef(np.nan_to_num(corr_files.sel(chromo='HbO',trial_type=trial_type).values).reshape(n_files,-1), rowvar=True)
            foo_hbr = np.corrcoef(np.nan_to_num(corr_files.sel(chromo='HbR',trial_type=trial_type).values).reshape(n_files,-1), rowvar=True)
            # foo_hbo = np.corrcoef(np.nan_to_num(corr_hbo_files[trial_type]), rowvar=True)
            # foo_hbr = np.corrcoef(np.nan_to_num(corr_hbr_files[trial_type]), rowvar=True)

            # compute the mean repeatability for HbO and HbR
            mean_hbo = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].mean()
            mean_hbr = foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].mean()

            da_hbo = xr.DataArray(
                mean_hbo,
                dims=["subj","trial_type","chromo"],
                coords={"trial_type": [trial_type], "chromo": ["HbO"], "subj": [curr_subj]}
            )
            da_hbr = xr.DataArray(
                mean_hbr,
                dims=["subj","trial_type","chromo"],
                coords={"trial_type": [trial_type], "chromo": ["HbR"], "subj": [curr_subj]}
            )
            if idx_trial_type == 0:
                repeatability_trial_type_mean = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                repeatability_trial_type_mean_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                repeatability_trial_type_mean = xr.concat([repeatability_trial_type_mean, repeatability_trial_type_mean_tmp], dim="trial_type")                

            # compute the std repeatability for HbO and HbR
            mean_hbo = foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].std()
            mean_hbr = foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].std()

            da_hbo = xr.DataArray(
                mean_hbo,
                dims=["subj","trial_type","chromo"],
                coords={"trial_type": [trial_type], "chromo": ["HbO"], "subj": [curr_subj]}
            )
            da_hbr = xr.DataArray(
                mean_hbr,
                dims=["subj","trial_type","chromo"],
                coords={"trial_type": [trial_type], "chromo": ["HbR"], "subj": [curr_subj]}
            )
            if idx_trial_type == 0:
                repeatability_trial_type_std = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                repeatability_trial_type_std_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                repeatability_trial_type_std = xr.concat([repeatability_trial_type_std, repeatability_trial_type_std_tmp], dim="trial_type")                

        if idx_subj == 0:
            repeatability_subj_mean = repeatability_trial_type_mean.copy()
            repeatability_subj_std = repeatability_trial_type_std.copy()
        else:
            repeatability_subj_mean = xr.concat([repeatability_subj_mean, repeatability_trial_type_mean], dim="subj")
            repeatability_subj_std = xr.concat([repeatability_subj_std, repeatability_trial_type_std], dim="subj")
    # end of subject loop

    return conc_ts_subjs, conc_var_subjs, corr_subj_files, repeatability_subj_mean, repeatability_subj_std



def get_correlation_matrices( conc_ts_subjs, conc_var_subjs, cfg_dataset, unique_trial_types ):

    corr_subj = {}
    corr_subj_var = {}

    # Loop over subjects
    for idx_subj, curr_subj in enumerate(cfg_dataset['subj_ids']):

        print(f'Correlation Matrices for SUBJECT {curr_subj}')

        # loop over trial types
        for idx_trial_type, trial_type in enumerate(unique_trial_types):

            # get the correlation matrix
            if 1:
                corr_hbo = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values, rowvar=True )
                corr_hbr = np.corrcoef( conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values, rowvar=True )
            else:
                corr_hbo = np.corrcoef( np.nan_to_num(conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbO').values,nan=0.0, posinf=0.0, neginf=0.0), rowvar=True )
                corr_hbr = np.corrcoef( np.nan_to_num(conc_ts_subjs[idx_subj][trial_type].sel(chromo='HbR').values,nan=0.0, posinf=0.0, neginf=0.0), rowvar=True )


            # get the correlation matrix for each subject
            da_hbo = xr.DataArray(
                corr_hbo.reshape(1,1,1,-1),
                dims=["subj", "trial_type", "chromo", "correlation"],
                coords={"chromo": ["HbO"], "trial_type": [trial_type], "subj": [curr_subj]}
            )
            da_hbr = xr.DataArray(
                corr_hbr.reshape(1,1,1,-1),
                dims=["subj", "trial_type", "chromo", "correlation"],
                coords={"chromo": ["HbR"], "trial_type": [trial_type], "subj": [curr_subj]}
            )
            if idx_trial_type == 0:
                da_chromo = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                da_chromo_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                da_chromo = xr.concat([da_chromo, da_chromo_tmp], dim="trial_type")

            # get the variance of each element in the correlation matrix
            da_hbo = xr.DataArray(
                (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbO').values[:, np.newaxis].T).reshape(1,1,1,-1),
                dims=["subj", "trial_type", "chromo", "correlation"],
                coords={"chromo": ["HbO"], "trial_type": [trial_type], "subj": [curr_subj]}
            )
            da_hbr = xr.DataArray(
                (conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis] + conc_var_subjs[idx_subj][trial_type].sel(chromo='HbR').values[:, np.newaxis].T).reshape(1,1,1,-1),
                dims=["subj", "trial_type", "chromo", "correlation"],
                coords={"chromo": ["HbR"], "trial_type": [trial_type], "subj": [curr_subj]}
            )
            if idx_trial_type == 0:
                da_chromo_var = xr.concat([da_hbo, da_hbr], dim="chromo")
            else:
                da_chromo_var_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
                da_chromo_var = xr.concat([da_chromo_var, da_chromo_var_tmp], dim="trial_type")
        # end of trial type loop

        if idx_subj == 0:
            corr_subj = da_chromo.copy()
            corr_subj_var = da_chromo_var.copy()
        else:
            corr_subj = xr.concat([corr_subj, da_chromo], dim="subj")
            corr_subj_var = xr.concat([corr_subj_var, da_chromo_var], dim="subj")
    # end of subject loop

    return corr_subj, corr_subj_var


def get_reliability(corr_subj, unique_trial_types):

    # get the reliability
    reliability_mean = {}
    reliability_std = {}
    for idx_trial_type, trial_type in enumerate(unique_trial_types):
        foo_hbo = np.corrcoef(np.nan_to_num(corr_subj.sel(chromo='HbO',trial_type=trial_type).values), rowvar=True)
        foo_hbr = np.corrcoef(np.nan_to_num(corr_subj.sel(chromo='HbR',trial_type=trial_type).values), rowvar=True)

        # get the mean reliability of the correlation matrix
        da_hbo = xr.DataArray(
            foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].mean().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbO"]}
        )
        da_hbr = xr.DataArray(
            foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].mean().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbR"]}
        )
        if idx_trial_type == 0:
            reliability_mean = xr.concat([da_hbo, da_hbr], dim="chromo")
        else:
            da_chromo_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
            reliability_mean = xr.concat([reliability_mean, da_chromo_tmp], dim="trial_type")

        # get the std reliability of the correlation matrix
        da_hbo = xr.DataArray(
            foo_hbo[np.triu_indices(foo_hbo.shape[0], k=1)].std().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbO"]}
        )
        da_hbr = xr.DataArray(
            foo_hbr[np.triu_indices(foo_hbr.shape[0], k=1)].std().reshape(1,1),
            dims=["trial_type","chromo"],
            coords={"trial_type":[trial_type],"chromo": ["HbR"]}
        )
        if idx_trial_type == 0:
            reliability_std = xr.concat([da_hbo, da_hbr], dim="chromo")
        else:
            da_chromo_tmp = xr.concat([da_hbo, da_hbr], dim="chromo")
            reliability_std = xr.concat([reliability_std, da_chromo_tmp], dim="trial_type")

    return reliability_mean, reliability_std


def group_correlation_matrices(corr_subj, corr_subj_var, unique_trial_types):
    # This is old... better to use the boot strapping method below

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
        corrz_hbo_subj = np.arctanh(corr_subj.sel(chromo='HbO',trial_type=trial_type).values)
        corrz_hbr_subj = np.arctanh(corr_subj.sel(chromo='HbR',trial_type=trial_type).values)

        # get the mean and std of the correlation matrix across subjects
        corrz_hbo_subj_mean[trial_type] = np.nanmean(corrz_hbo_subj, axis=0)
        corrz_hbr_subj_mean[trial_type] = np.nanmean(corrz_hbr_subj, axis=0)
        corrz_hbo_subj_std[trial_type] = np.nanstd(corrz_hbo_subj, axis=0)
        corrz_hbr_subj_std[trial_type] = np.nanstd(corrz_hbr_subj, axis=0)

        corrz_hbo_subj_meanweighted[trial_type] = np.nansum(corrz_hbo_subj * corr_subj_var.sel(chromo='HbO',trial_type=trial_type), axis=0) / np.nansum( corr_subj_var.sel(chromo='HbO',trial_type=trial_type), axis=0)
        corrz_hbr_subj_meanweighted[trial_type] = np.nansum(corrz_hbr_subj * corr_subj_var.sel(chromo='HbR',trial_type=trial_type), axis=0) / np.nansum( corr_subj_var.sel(chromo='HbR',trial_type=trial_type), axis=0)

        # get the mean and std of the correlation matrix across subjects
        corr_hbo_subj_mean[trial_type] = np.tanh(corrz_hbo_subj_mean[trial_type])
        corr_hbr_subj_mean[trial_type] = np.tanh(corrz_hbr_subj_mean[trial_type])
        corr_hbo_subj_std[trial_type] = np.tanh(corrz_hbo_subj_std[trial_type])
        corr_hbr_subj_std[trial_type] = np.tanh(corrz_hbr_subj_std[trial_type])

        corr_hbo_subj_meanweighted[trial_type] = np.tanh(corrz_hbo_subj_meanweighted[trial_type])
        corr_hbr_subj_meanweighted[trial_type] = np.tanh(corrz_hbr_subj_meanweighted[trial_type])

    return corr_hbo_subj_mean, corr_hbr_subj_mean, corr_hbo_subj_std, corr_hbr_subj_std, corr_hbo_subj_meanweighted, corr_hbr_subj_meanweighted



def boot_strap_corr(corr_subj, corr_subj_var,trial_type_list, trial_type_diff=None):

    z_boot_mean_hbo = {}
    z_boot_mean_hbr = {}
    z_boot_se_hbo = {}
    z_boot_se_hbr = {}
    r_boot_mean_hbo = {}
    r_boot_mean_hbr = {}

    if trial_type_diff is not None:
        if len(trial_type_diff) != 2:
            raise ValueError("trial_type_diff must contain exactly two trial types for difference calculation.")
            return        
        trial_type_diff_str = trial_type_diff[0] + '-' + trial_type_diff[1]
        trial_type_list = np.append( trial_type_list, trial_type_diff_str )
    else:
        trial_type_diff_str = 'none-none'

    for trial_type in trial_type_list:

        print(f"Bootstrapping for trial type '{trial_type}'")

        n_boot = 1000  # number of bootstrap samples

        n_subjects = len(corr_subj.subj)
        n_channels = int(np.sqrt(len(corr_subj.correlation)))

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
                if trial_type != trial_type_diff_str:
                    R = corr_subj.sel(chromo='HbO', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # R = corr_hbo_subj[trial_type][idx]
                    z = np.arctanh(R)  # Fisher z-transform
                    w = 1 / corr_subj_var.sel(chromo='HbO', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # w = 1 / corr_hbo_subj_var[trial_type][idx]
                else:
                    z = np.arctanh(corr_subj.sel(chromo='HbO', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        - corr_subj.sel(chromo='HbO', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)  # Fisher z-transform
                    w = 1 / (corr_subj_var.sel(chromo='HbO', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        + corr_subj_var.sel(chromo='HbO', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)
                    # z = np.arctanh(corr_hbo_subj[trial_type_diff[0]][idx] - corr_hbo_subj[trial_type_diff[1]][idx])  # Fisher z-transform
                    # w = 1 / (corr_hbo_subj_var[trial_type_diff[0]][idx] + corr_hbo_subj_var[trial_type_diff[1]][idx])
                z_weighted_sum_hbo += np.nan_to_num(z / w) # turn nan to 0
                weight_sum_hbo += np.nan_to_num(1 / w) # turn nan to 0

                if trial_type != trial_type_diff_str:
                    R = corr_subj.sel(chromo='HbR', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # R = corr_hbr_subj[trial_type][idx]
                    z = np.arctanh(R)  # Fisher z-transform
                    w = 1 / corr_subj_var.sel(chromo='HbR', trial_type=trial_type, subj=corr_subj.subj[idx]).values
                    # w = 1 / corr_hbr_subj_var[trial_type][idx]
                else:
                    z = np.arctanh(corr_subj.sel(chromo='HbR', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        - corr_subj.sel(chromo='HbR', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)
                    w = 1 / (corr_subj_var.sel(chromo='HbR', trial_type=trial_type_diff[0], subj=corr_subj.subj[idx]).values \
                        + corr_subj_var.sel(chromo='HbR', trial_type=trial_type_diff[1], subj=corr_subj.subj[idx]).values)
                    # z = np.arctanh(corr_hbr_subj[trial_type_diff[0]][idx] - corr_hbr_subj[trial_type_diff[1]][idx])                z_weighted_sum_hbr += np.nan_to_num(z / w) # turn nan to 0
                    # w = 1 / (corr_hbr_subj_var[trial_type_diff[0]][idx] + corr_hbr_subj_var[trial_type_diff[1]][idx])
                z_weighted_sum_hbr += np.nan_to_num(z / w)
                weight_sum_hbr += np.nan_to_num(1 / w) # turn nan to 0

            z_boot_samples_hbo[b] = z_weighted_sum_hbo / weight_sum_hbo
            z_boot_samples_hbr[b] = z_weighted_sum_hbr / weight_sum_hbr

        # Mean of bootstrap samples
        # Store mean of bootstrap samples in a single xarray.DataArray named z_boot_mean
        if trial_type == trial_type_list[0]:
            z_boot_mean = xr.DataArray(
                np.stack([
                    np.mean(z_boot_samples_hbo, axis=0),
                    np.mean(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation"],
                coords={"chromo": ["HbO", "HbR"]}
            )
            z_boot_mean = z_boot_mean.expand_dims(dim={"trial_type": [trial_type]})
        else:
            new = xr.DataArray(
                np.stack([
                    np.mean(z_boot_samples_hbo, axis=0),
                    np.mean(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation"],
                coords={"chromo": ["HbO", "HbR"]}
            )
            new = new.expand_dims(dim={"trial_type": [trial_type]})
            z_boot_mean = xr.concat([z_boot_mean, new], dim="trial_type")

        # Standard error (SE)
        # Standard error (SE) as xarray.DataArray with trial_type, chromo, correlation dims
        if trial_type == trial_type_list[0]:
            z_boot_se = xr.DataArray(
                np.stack([
                    np.std(z_boot_samples_hbo, axis=0),
                    np.std(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation"],
                coords={"chromo": ["HbO", "HbR"]}
            )
            z_boot_se = z_boot_se.expand_dims(dim={"trial_type": [trial_type]})
        else:
            new_se = xr.DataArray(
                np.stack([
                    np.std(z_boot_samples_hbo, axis=0),
                    np.std(z_boot_samples_hbr, axis=0)
                ]),
                dims=["chromo", "correlation"],
                coords={"chromo": ["HbO", "HbR"]}
            )
            new_se = new_se.expand_dims(dim={"trial_type": [trial_type]})
            z_boot_se = xr.concat([z_boot_se, new_se], dim="trial_type")

        # Confidence intervals (e.g. 95%)
        z_ci_lower_hbo = np.percentile(z_boot_samples_hbo, 2.5, axis=0)
        z_ci_upper_hbo = np.percentile(z_boot_samples_hbo, 97.5, axis=0)
        z_ci_lower_hbr = np.percentile(z_boot_samples_hbr, 2.5, axis=0)
        z_ci_upper_hbr = np.percentile(z_boot_samples_hbr, 97.5, axis=0)

        # Convert back to r-space if needed
        r_boot_mean = np.tanh(z_boot_mean)
        # r_boot_mean_hbr[trial_type] = np.tanh(z_boot_mean_hbr[trial_type])
        # r_ci_lower_hbo = np.tanh(z_ci_lower_hbo)
        # r_ci_upper_hbo = np.tanh(z_ci_upper_hbo)
        # r_ci_lower_hbr = np.tanh(z_ci_lower_hbr)
        # r_ci_upper_hbr = np.tanh(z_ci_upper_hbr)

    return z_boot_mean, z_boot_se, r_boot_mean

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.io import loadmat
import glob
import h5py
from scipy.stats import binned_statistic
from collections import namedtuple
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

from ..utility.utils import bandpass_filter
from ..utility.utils import interpolate_signal_with_nans
from ..utility.utils import FlightClusters


class Data_Prep_for_Raw():
    """
    Data preprocessing for the raw data, before transforming to torch tensor data.

    Example code:
        parent_folder = '/mnt/at-globusshare/Users/Yongrong/Dataset_3_14543_240419'
        su_file_path = '/mnt/scratch09/yongrong/datahub/bat-intelligence/out_good_units_Dataset_3_14543_240419.mat'
        use_IMU_flag = True,
        use_click_flag = False,
        use_LFP_flag = True,
        use_ripple_flag = False,
        only_flight_flag = True,
        bin_width = 0.010 # 10 ms = 0.01 seconds, so the new sampling frequency is 100 Hz

        data_prep_for_raw = Data_Prep_for_Raw(
            parent_folder = parent_folder,
            su_file_path = su_file_path,
            use_IMU_flag = use_IMU_flag,
            use_click_flag = use_click_flag,
            use_LFP_flag = use_LFP_flag,
            use_ripple_flag = use_ripple_flag,
            only_flight_flag = only_flight_flag,
            bin_width = bin_width, 
        )
        all_behavior = data_prep_for_raw.get_all_behavior()
        all_spike = data_prep_for_raw.get_all_spike()
        spike_ratio = data_prep_for_raw.spike_ratio
    """
    def __init__(
        self,
        parent_folder,
        su_file_path,  
        use_speed_head_dir, # whether to use speed and head direction or not
        use_IMU_flag,
        use_click_flag,
        use_LFP_flag,
        use_ripple_flag,
        only_flight_flag,
        no_3d_location=False, # default: we use 3d locations
        spike_count_flag=False, # default: we use spiking event, not spike counts 
        bin_width=0.010, # 10 ms = 0.01 seconds, so the new sampling frequency is 100 Hz
        flight_cluster_alpha=0.8, # alpha for flight clustering
        split_flights_for_train: str = 'manual', # the way to split flight paths for train + valid + test; either 'manual' or 'random'
        type_train_num=1,  # for splitting flight clustering data
        type_valid_num=1,  # for splitting flight clustering data
        special_id_count_thresh=3,  # for splitting flight clustering data
        only_use_special_id_4train_flag=False, # Only use the "special_id" flight types (e.g., flight clusters with more than "special_id_count_thresh" repeats) for training 
        simulate_place_cell_flag=False, # Whether to generate simulation data of pure place cells using training data.
        # After splitting fight-clustering data into train, validation and test, we may want to use/split each flight trajectory as/into one/multiple data samples.
        # With such data, we can build a network (such as Transformer) to model the temporal dependency of animal behavior (and neural activity) during each flight 
        split_each_trajectory_flag=False, 
        window_size: int|None = None, # for splitting each trajectory into multiple samples
        stride_train: int|None = None, # splitting each trajectory into multiple samples, for training data
        stride_test: int|None = None, # splitting each trajectory into multiple samples, for test data
    ):
        super().__init__()
        self.parent_folder = parent_folder
        self.su_file_path = glob.glob(su_file_path)  # a list of strings
        self.use_speed_head_dir = use_speed_head_dir
        self.use_IMU_flag = use_IMU_flag
        self.use_click_flag = use_click_flag
        self.use_LFP_flag = use_LFP_flag
        self.use_ripple_flag = use_ripple_flag
        self.only_flight_flag = only_flight_flag
        self.no_3d_location = no_3d_location
        self.spike_count_flag = spike_count_flag
        self.type_train_num = type_train_num
        self.type_valid_num = type_valid_num
        self.special_id_count_thresh = special_id_count_thresh
        self.only_use_special_id_4train_flag = only_use_special_id_4train_flag
        self.simulate_place_cell_flag = simulate_place_cell_flag
        self.split_each_trajectory_flag = split_each_trajectory_flag
        self.window_size = window_size
        self.stride_train = stride_train
        self.stride_test = stride_test

        # data resampling
        self.bin_width = bin_width
        self.global_t_start = 0.0 + bin_width / 2
        self.global_t_end = None
        self.all_time = None  # for 3D location, IMU
        self.bin_edges = None  # for spike counts, LFP 

        self.r_lim = None # boundaries of the room/outdoor arena
        self.all_behavior = None
        self.all_spike = None
        self.flight_ind = None
        self.all_time_flight = None # after interpolation (binning), the time of each data point during flight
        self.spike_ratio = None # spike ratio for each unit after resampling

        self.flight_cluster_alpha = flight_cluster_alpha
        self.flight_cluster_results = None
        self.all_behavior_clustered = None  # a list of arrays, each array contains behavior for each trajectory
        self.all_behavior_type = None  # a list of trajectory clustering types
        self.all_spike_clustered = None  # a list of arrays, each array contains neuronal activity for each trajectory
        self.max_len_of_flight_trajectory = None # maximum length of all flight trajectories in this session
        self.split_flights_for_train = split_flights_for_train
        self.flight_type_test_indomain = None
        self.flight_type_test_ood = None

        # after flight clustering, split the trajectories for training, evaluation and test
        self.samples_train = None
        self.samples_valid = None
        self.samples_test = None

        # internal state flag to enforce call order inside the class
        self._behavior_ready = False  
        self._spike_ready = False
        self._flight_cluster_ready = False

    def get_all_behavior(self):
        """
        Get all the behavior information, including 3D locations, flight status, IMU of the head (if use_IMU_flag),
        click (if use_click_flag), average of the best LFP (if use_LFP_flag), ripple (if use_ripple_flag).
        If only_flight_flag, we only use the data during flight.
        """
        # behavior
        bhvFile = glob.glob(self.parent_folder + '/Extracted_Behavior_*.mat')[0]
        # the max of the speed is about 6 meters per second in the flight rooom # just apply a median filter
        bhv = loadmat(bhvFile)
        # print (f'bhv.keys(): {bhv.keys()}')
        # print (f"bhv['Fs']: {bhv['Fs']}, bhv['bflying'].shape: {bhv['bflying'].shape}, bhv['v_abs'].shape: {bhv['v_abs'].shape}, bhv['r'].shape: {bhv['r'].shape}")
        # bhv['Fs']: [[120]], bhv['bflying'].shape: (642724, 1), bhv['v_abs'].shape: (642724, 1), bhv['r'].shape: (642724, 3)

        # bhv_thresh = 6.0
        bhv_bflying = bhv['bflying'][:,0] # 'v_abs'
        bhv_v_abs = bhv['v_abs'][:,0]
        # bhv_v_abs[np.where(bhv_v_abs > bhv_thresh)] = bhv_thresh
        bhv_r = bhv['r'] # 3D position
        if 'Dataset_2' in self.parent_folder:
            bhv_t = bhv['t'][:,0]
        else:
            bhv_t = bhv['t'][0,:] # np.arange(len(bhv_v_abs))/bhv['Fs'][0]
        # boundaries of the room/outdoor arena
        self.r_lim = bhv['r_lim']

        # IMU
        if self.use_IMU_flag:
            imuFile = glob.glob(self.parent_folder + '/IMU_data.mat')[0]
            imu = loadmat(imuFile)
            NP_imu = imu['NP_imu']
            imu_acc = NP_imu[0][0]['acc']
            imu_gyr = NP_imu[0][0]['gyr']
            imu_t = NP_imu[0][0]['t'][0,:] # np.arange(len(imu_acc))/NP_imu[0][0][3][0,0]
            # print (f'imu_acc.shape: {imu_acc.shape}, imu_gyr.shape: {imu_gyr.shape}, imu_t.shape: {imu_t.shape}')
            # imu_acc.shape: (2633948, 3), imu_gyr.shape: (2633948, 3), imu_t.shape: (2633948,)

        # echolocation click
        if self.use_click_flag:
            clickFile = glob.glob(self.parent_folder + '/Detected_Clicks_*.mat')[0]
            click = loadmat(clickFile)
            # print (f'click.keys(): {click.keys()}')

            Detected_Clicks = click['Detected_Clicks']
            click_time = Detected_Clicks[0][0][0][:,0]
            click_shape = Detected_Clicks[0][0][1]
            click_amp = Detected_Clicks[0][0][2][:,0]
            click_shape_t_ms = np.arange(click_shape.shape[1])/Detected_Clicks[0][0][4][0,0] * 1000
            # print (f'click_time.shape: {click_time.shape}, click_shape.shape: {click_shape.shape}, click_amp.shape: {click_amp.shape}')
            # click_time.shape: (1444,), click_shape.shape: (1444, 96), click_amp.shape: (1444,)

        if self.use_LFP_flag:
            bpcFile = glob.glob(self.parent_folder + '/BPC_Dataset_*.mat')[0]
            bpc = loadmat(bpcFile)
            # print (f'bpc.keys(): {bpc.keys()}')
            # so we use the 2 nearest neighbours of the channels (i.e. from channel 39 - 43)
            # then average across these 5 channels
            # be careful, some channels may have artifacts (2-second blanck) for the LFPs
            lfpFile = glob.glob(self.parent_folder + '/LFP_probe*.mat')

            probeind = bpc['best_prb_ch'][0][0] - 1 # matlab starts from 1 but python start from 0
            file_path = lfpFile[probeind]

            with h5py.File(file_path, 'r') as ff:
                # print("Top-level keys:")
                # for key in ff.keys():
                #     print(key)
                red_out = ff['red_out']
                # print("red_out type:", type(red_out))
                # print("red_out keys (if group):", list(red_out.keys()))  # If it's a group
                
                # Load LFP matrix
                lfp_data = red_out['lfp'][:]  # shape: (n_channels, n_timepoints) or similar
                # print("lfp_data shape:", lfp_data.shape)
                # Load timestamps 
                lfp_t_ds = red_out['t_ds'][:][:,0] # /red_out['sampling_freq'][:][0,0]
                # print("lfp_t_ds shape:", lfp_t_ds.shape)
                lfp_sampling_freq = red_out['sampling_freq'][:][0,0]
                # print (f'lfp_sampling_freq: {lfp_sampling_freq}')
                lfp_voltage_scaling = red_out['voltage_scaling'][:]
                # print (f'lfp_voltage_scaling.shape: {lfp_voltage_scaling.shape}')

                # # Channel map and positions if needed
                # channel_map = red_out['channelMap'][:]
                # channel_positions = red_out['channelPositions'][:]

            tempinds = [-2, -1, 0, 1, 2] + bpc['best_prb_ch'][0][1] - 1
            # tempinds
            lfp_data_best_ave = np.mean(lfp_data[tempinds, :], axis=0)

            fs = np.copy(lfp_sampling_freq)  # sampling rate in Hz
            lfp = np.copy(lfp_data_best_ave)  # 1D NumPy array, e.g., from one channel
            # theta_band = (6, 10)  # Hz
            beta_band = (12, 30) # Hz
            # bin_size_sec = 0.1  # 100 ms
            lfp_filter = bandpass_filter(lfp, fs, beta_band)

        # time resampling
        self.global_t_end = np.min([bhv_t[-1], imu_t[-1]]) if self.use_IMU_flag else bhv_t[-1]
        # print (f'global_t_end: {global_t_end}')
        all_time =np.arange(self.global_t_start, self.global_t_end, self.bin_width)
        # bin spike times
        bin_edges = np.append(all_time, all_time[-1] + self.bin_width) - self.bin_width/2
        if self.global_t_end < bin_edges[-1]: # bin_edges out of the range
            all_time = all_time[:-1]
            bin_edges = bin_edges[:-1]
        self.all_time = all_time
        self.bin_edges = bin_edges

        # behavior resampling
        all_behavior = []
        for ii in range(bhv_r.shape[1]):
            temp = interpolate_signal_with_nans(bhv_t, bhv_r[:,ii], self.all_time)
            # print (f'temp.shape: {temp.shape}')
            all_behavior.append(temp)
        # add speed and head direction
        if self.use_speed_head_dir:
            traj = np.array(all_behavior[:bhv_r.shape[1]]).T # shape: (time, 3)
            vel = np.gradient(traj, self.bin_width, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            # vel = np.insert(np.diff(traj, axis=0), 0, 0.0, axis=0)/self.bin_width
            # speed = np.linalg.norm(vel, axis=1, keepdims=True)
            vel_unit = vel / (speed[:,None] + 1e-8)
            # print (f'speed.shape: {speed.shape}')
            all_behavior.append(speed)
            for ii in range(vel_unit.shape[1]):
                # print (f'vel_unit[:, ii].shape: {vel_unit[:, ii].shape}')
                all_behavior.append(vel_unit[:, ii])
        # flight status
        bhv_bflying_resample = interpolate_signal_with_nans(bhv_t, bhv_bflying, self.all_time)
        bhv_v_abs_resample = interpolate_signal_with_nans(bhv_t, bhv_v_abs, self.all_time)
        # print(f'np.mean(np.abs(bhv_v_abs_resample - speed)): {np.mean(np.abs(bhv_v_abs_resample - speed))}')
        # print (f'temp.shape: {temp.shape}')
        # all_behavior.append(temp)

        if self.use_IMU_flag:
            for ii in range(imu_acc.shape[1]):
                temp = interpolate_signal_with_nans(imu_t, imu_acc[:,ii], self.all_time)
                # print (f'temp.shape: {temp.shape}')
                all_behavior.append(temp)
            for ii in range(imu_gyr.shape[1]):
                temp = interpolate_signal_with_nans(imu_t, imu_gyr[:,ii], self.all_time)
                # print (f'temp.shape: {temp.shape}')
                all_behavior.append(temp)

        if self.use_click_flag:
            click_counts, _ = np.histogram(click_time, bins=self.bin_edges)
            all_behavior.append(click_counts)

        if self.use_LFP_flag:
            # lfp_filter, beta_band: Compute mean power in each bin (vectorized); and align
            binned_power, _, _ = binned_statistic(
                lfp_t_ds,                   # Time stamps
                lfp_filter ** 2,            # Power
                statistic='mean',
                bins=self.bin_edges,        # Bin edges
                range=(self.bin_edges[0], self.bin_edges[-1]),
            )
            all_behavior.append(binned_power)
            
        all_behavior = np.array(all_behavior)
        # print (f'all_behavior.shape: {all_behavior.shape}') # all_behavior.shape: (11, 524859)

        # only use the flight data
        if self.only_flight_flag:
            self.flight_ind = np.where(bhv_bflying_resample>0.0)[0]
            self.all_time_flight = self.all_time[self.flight_ind]
            all_behavior = all_behavior[:, self.flight_ind]
            # all_spike = all_spike[:, flight_ind]
            # remove flying dimension from the all_behavior
            # all_behavior = np.delete(all_behavior, 3, axis=0)
            # print (f'all_behavior.shape: {all_behavior.shape}') 

        # not use 3d location
        if self.no_3d_location:
            all_behavior = np.delete(all_behavior, [0,1,2], axis=0)
            # print (f'all_behavior.shape: {all_behavior.shape}') 

        self.all_behavior = all_behavior

        self._behavior_ready = True
        
        return self.all_behavior

    def get_all_spike(self):
        """
        Get the single-unit spikes 
        """
        if not self._behavior_ready:
            raise RuntimeError("You must run get_all_behavior() before get_all_spike().")

        su_goods_cluster_id = []
        su_goods_time = []
        for temppath in self.su_file_path:
            su_good = loadmat(temppath)
            temp_cluster_id = np.array([x[0][0,0] for x in su_good['out_Gunits']['cluster_id']])
            temp_good1_time = [x[0][:,0]/1e6 for x in su_good['out_Gunits']['spikeTimes_usec']]
            su_goods_cluster_id.append(temp_cluster_id)
            su_goods_time.append(temp_good1_time)

        # merge units
        mergesFile = glob.glob(self.parent_folder + '/merges_Dataset_*.mat')[0]
        merges = loadmat(mergesFile)
        # here represents three probes; the first probe does not have any units need to be merged
        # for the second probe, the cluster_ids  of 170 and 176 needs to be merged
        # for the third probe, the cluster_ids of 47, 66, 74 need to be merged
        # we use refractory period of 1 ms to merge two very closed spikes 

        refractory = 0.001 # refractory period of 1 ms
        probe_num = len(self.su_file_path) # three probes
        for ii in range(probe_num):
            if len(merges['merges'][ii][0]):
                cluster_ids = merges['merges'][ii][0][0][0][0] # # array of unit IDs to merge
                # print (f'Probe : {ii}, cluster_ids: {cluster_ids}')
                # Select the correct list of cluster IDs and spike times for this probe
                cluster_id_list = su_goods_cluster_id[ii]
                time_list = su_goods_time[ii]
                # Find indices of the units to be merged in the current probe
                merge_indices = [np.where(np.array(cluster_id_list) == cid)[0][0] for cid in cluster_ids]
                # Gather all spike times from the units to be merged
                merged_times = np.concatenate([time_list[idx] for idx in merge_indices])
                merged_times.sort()
                # Remove spikes that occur within the refractory period
                # Keep the first spike, then only keep the next if it's beyond the refractory window
                clean_merged_times = [merged_times[0]]
                for t in merged_times[1:]:
                    if t - clean_merged_times[-1] > refractory:
                        clean_merged_times.append(t)
                clean_merged_times = np.array(clean_merged_times)
                # Optionally, replace the first unit's times with the merged version and
                # remove the others or mark them as merged
                time_list[merge_indices[0]] = clean_merged_times
                for idx in merge_indices[1:]:
                    time_list[idx] = np.array([])  # mark as empty/merged
                # You may also want to log this
                # print(f"Merged units {cluster_ids} into unit {cluster_ids[0]} with {len(clean_merged_times)} spikes remaining.")
                # remove the merged-away empty units from your lists (su_goods_time and su_goods_cluster_id)
                # Keep only those entries that have spike times
                valid_indices = [i for i, t in enumerate(time_list) if len(t) > 0]
                su_goods_time[ii] = [time_list[i] for i in valid_indices]
                su_goods_cluster_id[ii] = [cluster_id_list[i] for i in valid_indices]

        # su_goods_time = [su_good1_time, su_good2_time, su_good3_time]
        all_spike = []
        for su_good_time in su_goods_time:
            for ii in range(len(su_good_time)):
                # Bin the spike times
                spike_counts, _ = np.histogram(su_good_time[ii], bins=self.bin_edges)
                all_spike.append(spike_counts)
        all_spike = np.array(all_spike)
        if not self.spike_count_flag:
            all_spike = np.where(all_spike>0.0, 1.0, 0.0)
        # print (f'all_spike.shape: {all_spike.shape}') # all_spike.shape: (289, 524859)

        # only use the flight data
        if self.only_flight_flag:
            # flight_ind = np.where(self.all_behavior[3]>0.0)[0]
            # print (f'Ratio of flight: {len(flight_ind)/self.all_behavior.shape[1]}')
            all_spike = all_spike[:, self.flight_ind]
            # print (f'all_spike.shape: {all_spike.shape}')

        # spike ratio for each neuron
        self.spike_ratio = np.sum((all_spike>0) * 1.0, axis=1)/all_spike.shape[1]

        self.all_spike = all_spike

        self._spike_ready = True

        return self.all_spike
    
    def get_flight_cluster(self):
        """
        Perform flight clustering

        Plot intermediate flight clustering results:
            tempid = 9
            tempr = flight_cluster_results['pos'][:, :, np.where(flight_cluster_results['id']==tempid)[0]]
            fig, ax = plt.subplots(ncols=3, figsize=(8, 2))
            for ii in range(tempr.shape[2]):
                ax[0].plot(tempr[0,:,ii], tempr[1,:,ii])
                ax[0].set_xlabel('X')
                ax[0].set_ylabel('Y')
                ax[0].set_xlim(bhv['r_lim'][0])
                ax[0].set_ylim(bhv['r_lim'][1])
                ax[0].set_title(f'Cluster {tempid}; Raw data')
                ax[1].plot(tempr[0,:,ii], tempr[2,:,ii])
                ax[1].set_xlabel('X')
                ax[1].set_ylabel('Z')
                ax[1].set_xlim(bhv['r_lim'][0])
                ax[1].set_ylim(bhv['r_lim'][2])
                ax[2].plot(tempr[1,:,ii], tempr[2,:,ii])
                ax[2].set_xlabel('Y')
                ax[2].set_ylabel('Z')
                ax[2].set_xlim(bhv['r_lim'][1])
                ax[2].set_ylim(bhv['r_lim'][2])

            start_time = bhv_t[flight_cluster_results['start_frame'][np.where(flight_cluster_results['id']==tempid)[0]]]
            stop_time = bhv_t[flight_cluster_results['stop_frame'][np.where(flight_cluster_results['id']==tempid)[0]]]
            fig, ax = plt.subplots(ncols=3, figsize=(8, 2))
            for ii in range(tempr.shape[2]):
                tempind = np.where((all_time_flight>=start_time[ii]) & (all_time_flight<=stop_time[ii]))[0]
                ax[0].plot(all_behavior[0, tempind], all_behavior[1, tempind])
                ax[0].set_xlabel('X')
                ax[0].set_ylabel('Y')
                ax[0].set_xlim(bhv['r_lim'][0])
                ax[0].set_ylim(bhv['r_lim'][1])
                ax[0].set_title(f'After interpolation')
                ax[1].plot(all_behavior[0, tempind], all_behavior[2, tempind])
                ax[1].set_xlabel('X')
                ax[1].set_ylabel('Z')
                ax[1].set_xlim(bhv['r_lim'][0])
                ax[1].set_ylim(bhv['r_lim'][2])
                ax[2].plot(all_behavior[1, tempind], all_behavior[2, tempind])
                ax[2].set_xlabel('Y')
                ax[2].set_ylabel('Z')
                ax[2].set_xlim(bhv['r_lim'][1])
                ax[2].set_ylim(bhv['r_lim'][2])

            ncols = tempr.shape[2]
            vmin, vmax = np.min(all_spike), 10
            fig, ax = plt.subplots(ncols=ncols, figsize=(ncols*1.5, 2))
            for ii in range(ncols):
                tempind = np.where((all_time_flight>=start_time[ii]) & (all_time_flight<=stop_time[ii]))[0]
                im = ax[ii].matshow(all_spike[:, tempind], vmin=vmin, vmax=vmax, cmap='hot_r')
                ax[ii].set_aspect(0.15)
                ax[ii].set_title(f'Repeat {ii}')
            ax[0].set_xlabel('Time frame')
            ax[0].set_ylabel('Unit')
            plt.colorbar(im, ax=ax[ii], label="spikes/s")
        """
        if not self._spike_ready:
            raise RuntimeError('You must run get_all_behavior() and get_all_spike() before get_flight_cluster().')
        
        bhvFile = glob.glob(self.parent_folder + '/Extracted_Behavior_*.mat')[0]
        bhv = loadmat(bhvFile)
        if 'Dataset_2' in self.parent_folder:
            bhv_t = bhv['t'][:,0]
        else:
            bhv_t = bhv['t'][0,:] 

        # Cluster Flights (see Forli and Yartsev, Nature 2023 for details on the hierarchical clustering procedure)
        # Similar to Line 196 in NP_Bat_Demo.m
        if 'Dataset_2' in self.parent_folder: # More liberal clustering for Field Station Dataset
            self.flight_cluster_alpha = 1.2
        flight_clusters = FlightClusters(r_vector=bhv['r'], f_vector=bhv['bflying'][:,0], Fs=bhv['Fs'][0][0], frechet_clus=True, alpha=self.flight_cluster_alpha)
        self.flight_cluster_results = flight_clusters.compute()
        unique_ids = np.unique(self.flight_cluster_results['id'])
        while len(unique_ids)==1 and self.flight_cluster_alpha<1.5: # Try increasing the clustering param if only 1 cluster
            self.flight_cluster_alpha = self.flight_cluster_alpha + 0.1
            flight_clusters = FlightClusters(r_vector=bhv['r'], f_vector=bhv['bflying'][:,0], Fs=bhv['Fs'][0][0], frechet_clus=True, alpha=self.flight_cluster_alpha)
            self.flight_cluster_results = flight_clusters.compute()
            unique_ids = np.unique(self.flight_cluster_results['id'])

        # for each flight cluster type, we only use one repeat to train and evaluate the model
        # TODO: for each flight cluster type, we use the mean of the repeats to evaluate the model
        all_behavior_clustered = []
        all_spike_clustered = []
        all_behavior_type = []
        for tempid in unique_ids:
            start_time = bhv_t[self.flight_cluster_results['start_frame'][np.where(self.flight_cluster_results['id']==tempid)[0]]]
            stop_time = bhv_t[self.flight_cluster_results['stop_frame'][np.where(self.flight_cluster_results['id']==tempid)[0]]]
            for ii in range(len(start_time)):
                tempind = np.where((self.all_time_flight>=start_time[ii]) & (self.all_time_flight<=stop_time[ii]))[0]
                if len(tempind): # exclude the flight behavior beyond all_time_flight, e.g., at the very beginning of the session
                    all_behavior_clustered.append(self.all_behavior[:, tempind])
                    all_behavior_type.append(tempid)
                    all_spike_clustered.append(self.all_spike[:, tempind])
    
        # # shuffle all flights
        # np.random.seed(100)
        # np.random.shuffle(all_behavior_clustered)
        # np.random.shuffle(all_spike_clustered)

        self.all_behavior_clustered = all_behavior_clustered
        self.all_behavior_type = all_behavior_type
        self.all_spike_clustered = all_spike_clustered
        # print(f'len(all_behavior_clustered): {len(all_behavior_clustered)}, all_behavior_clustered[0].shape: {all_behavior_clustered[0].shape}') 
        # print(f'len(all_behavior_type):    {len(all_behavior_type)}, all_behavior_type: {all_behavior_type}')
        # print(f'len(all_spike_clustered):    {len(all_spike_clustered)}, all_spike_clustered[0].shape: {all_spike_clustered[0].shape}')
        # len(all_behavior_clustered): 37, all_behavior_clustered[0].shape: (3, 59)
        # len(all_behavior_type):    37,    all_behavior_type: [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]
        # len(all_spike_clustered):    37,    all_spike_clustered[0].shape: (289, 59)

        self.max_len_of_flight_trajectory = np.max([beh.shape[1] for beh in all_behavior_clustered])
        # print(f'max_len_of_flight_trajectory: {self.max_len_of_flight_trajectory}')

        self._flight_cluster_ready = True
        
        return self.all_behavior_clustered, self.all_behavior_type, self.all_spike_clustered
    
    def split_flight_cluster(self):
        """
        After performing flight clustering, we split trajectories for training, validation and test
        """
        if not self._flight_cluster_ready:
            raise RuntimeError('You must run get_flight_cluster() before running split_flight_cluster().')
        
        behavior_train, behavior_valid, behavior_test = [], [], []
        spike_train, spike_valid, spike_test = [], [], []
        flight_path_index_train, flight_path_index_valid, flight_path_index_test = [], [], [] # index of individual flight path
        flight_type_index_train, flight_type_index_valid, flight_type_index_test = [], [], [] # index of each flight cluster type

        # After flight clustering, some types have multiple flight paths while others have only one trajectory
        # There are several ways to split the data for modeling.

        if self.split_flights_for_train == 'manual':
            # We use some types to train the model and other types to validate and test the model. One way is to
            # use only one repeat from all types.
            # Or: we use all types to train the model and use the other repeats of certain types to validate and 
            # test the model
            # for each special_id, we use 'type_train_num' repeat for training, 'type_valid_num' for validation, others for test
            assert self.special_id_count_thresh > self.type_train_num + self.type_valid_num
            unique_ids, id_counts = np.unique(self.all_behavior_type, return_counts=True)
            # print(f'unique_ids: {unique_ids}')  # unique_ids: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 15 16 17 18 19 20 21]
            # print(f'id_counts: {id_counts}')  # id_counts: [2 1 9 1 1 1 1 1 8 2 1 1 1 1 1 1 1 1 1 1]
            # special_ids would be used for model training, validation and test
            special_ids = unique_ids[np.where(id_counts>=(self.special_id_count_thresh ))[0]]
            # print(f'special_ids: {special_ids}')  # special_ids: [3 9]

            if not self.only_use_special_id_4train_flag:
                for ii in range(len(self.all_behavior_type)):
                    if self.all_behavior_type[ii] not in special_ids:
                        behavior_train.append(self.all_behavior_clustered[ii])
                        spike_train.append(self.all_spike_clustered[ii])
                        flight_path_index_train.append(ii)
                        flight_type_index_train.append(self.all_behavior_type[ii])
            for unique_id in special_ids:
                tempinds = np.where(np.array(self.all_behavior_type)==unique_id)[0]
                for jj in range(len(tempinds)):
                    if jj < self.type_train_num:
                        behavior_train.append(self.all_behavior_clustered[tempinds[jj]])
                        spike_train.append(self.all_spike_clustered[tempinds[jj]])
                        flight_path_index_train.append(tempinds[jj])
                        flight_type_index_train.append(self.all_behavior_type[tempinds[jj]])
                    elif jj < self.type_train_num + self.type_valid_num:
                        behavior_valid.append(self.all_behavior_clustered[tempinds[jj]])
                        spike_valid.append(self.all_spike_clustered[tempinds[jj]])
                        flight_path_index_valid.append(tempinds[jj])
                        flight_type_index_valid.append(self.all_behavior_type[tempinds[jj]])
                    else:
                        behavior_test.append(self.all_behavior_clustered[tempinds[jj]])
                        spike_test.append(self.all_spike_clustered[tempinds[jj]])
                        flight_path_index_test.append(tempinds[jj])
                        flight_type_index_test.append(self.all_behavior_type[tempinds[jj]])

        elif self.split_flights_for_train == 'random':
            # Or we treat each flight path as a random data sample
            # We randomly select 70% for training, 15% for validation and 15% for test

            # len(all_behavior_clustered): 37, all_behavior_clustered[0].shape: (3, 59)
            # len(all_behavior_type):    37,    all_behavior_type: [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]
            # len(all_spike_clustered):    37,    all_spike_clustered[0].shape: (289, 59)
            np.random.seed(166)
            inds = np.random.choice(len(self.all_behavior_type), len(self.all_behavior_type), replace=False)
            train_num, valid_num = int(len(inds) * 0.7), int(len(inds) * 0.15)
            test_num = len(inds) - train_num - valid_num
            for ii in range(len(inds)):
                if ii < train_num:
                    behavior_train.append(self.all_behavior_clustered[inds[ii]])
                    spike_train.append(self.all_spike_clustered[inds[ii]])
                    flight_path_index_train.append(inds[ii])
                    flight_type_index_train.append(self.all_behavior_type[inds[ii]])
                elif ii >= train_num and ii < train_num + valid_num:
                    behavior_valid.append(self.all_behavior_clustered[inds[ii]])
                    spike_valid.append(self.all_spike_clustered[inds[ii]])
                    flight_path_index_valid.append(inds[ii])
                    flight_type_index_valid.append(self.all_behavior_type[inds[ii]])
                elif ii >= train_num + valid_num:
                    behavior_test.append(self.all_behavior_clustered[inds[ii]])
                    spike_test.append(self.all_spike_clustered[inds[ii]])
                    flight_path_index_test.append(inds[ii])
                    flight_type_index_test.append(self.all_behavior_type[inds[ii]])

        # For flight cluster types in test data, some of them also exist in the traininig or validation data, we call them 'flight_type_test_indomain'
        # Other types are only present in the test data, which are referred to as 'flight_type_test_ood'
        self.flight_type_test_indomain = list(set(flight_type_index_test) & (set(flight_type_index_train) | set(flight_type_index_valid)))
        self.flight_type_test_ood = list(set(flight_type_index_test) - (set(flight_type_index_train) | set(flight_type_index_valid)))
        # print(f'flight_type_index_train: {flight_type_index_train}')
        # print(f'flight_type_index_valid: {flight_type_index_valid}')
        # print(f'flight_type_index_test: {flight_type_index_test}')
        # print(f'self.flight_type_test_indomain: {self.flight_type_test_indomain}')
        # print(f'self.flight_type_test_ood: {self.flight_type_test_ood}')

        # Generate simulation data of "pure place-field cells" based on the actual behavior and rate maps computed from training data
        if self.simulate_place_cell_flag:
            loc_bin_width = 0.1 
            place_cell_simulator_3D = PlaceCellSimulator3D(bin_size=loc_bin_width, smoothing_sigma=1) # smoothing_sigma=1
            _ = place_cell_simulator_3D.fit(behavior_train, spike_train, dt=self.bin_width)
            spike_train_simulation = place_cell_simulator_3D.simulate(behavior_train, dt=self.bin_width)
            spike_valid_simulation = place_cell_simulator_3D.simulate(behavior_valid, dt=self.bin_width)
            spike_test_simulation = place_cell_simulator_3D.simulate(behavior_test, dt=self.bin_width)
            # # for sanity check
            # plot_2d_trajectory_4_simulation(
            #     behavior_train,
            #     spike_train,
            #     behavior_train,
            #     spike_train_simulation,
            #     behavior_test,
            #     spike_test,
            #     behavior_test,
            #     spike_test_simulation,
            #     units = [101, 130, 184], # For 'Dataset_3_29959_240403'
            #     loc_bin_width = loc_bin_width,
            #     dt = self.bin_width,
            # )
            spike_train = spike_train_simulation[:]
            spike_valid = spike_valid_simulation[:]
            spike_test = spike_test_simulation[:]

        # print(len(behavior_train), len(behavior_valid), len(behavior_test))  # 22 2 13
        # print(len(spike_train), len(spike_valid), len(spike_test))  # 22 2 13
        # shuffle the training trajectories
        np.random.seed(100)
        np.random.shuffle(behavior_train)
        np.random.seed(100)
        np.random.shuffle(spike_train)
        np.random.seed(100)
        np.random.shuffle(flight_path_index_train)
        np.random.seed(100)
        np.random.shuffle(flight_type_index_train)
        if self.split_each_trajectory_flag:
            # self.samples_train = (np.hstack(behavior_train), np.hstack(spike_train))
            # self.samples_valid = (np.hstack(behavior_valid), np.hstack(spike_valid))
            # self.samples_test = (np.hstack(behavior_test), np.hstack(spike_test))
            # print(f'Maximum flight length: {np.max([temp.shape[1] for temp in self.all_behavior_clustered])}')
            if self.window_size is None and self.stride_train is None and self.stride_test is None: 
                self.samples_train = self.split_each_trajectory(
                    behavior_train, 
                    spike_train, 
                    flight_path_index_train,
                    flight_type_index_train,
                    self.max_len_of_flight_trajectory, 
                    self.max_len_of_flight_trajectory
                )
                self.samples_valid = self.split_each_trajectory(
                    behavior_valid, 
                    spike_valid, 
                    flight_path_index_valid,
                    flight_type_index_valid,
                    self.max_len_of_flight_trajectory, 
                    self.max_len_of_flight_trajectory
                )
                self.samples_test = self.split_each_trajectory(
                    behavior_test, 
                    spike_test, 
                    flight_path_index_test,
                    flight_type_index_test,
                    self.max_len_of_flight_trajectory, 
                    self.max_len_of_flight_trajectory
                )
            else:
                self.samples_train = self.split_each_trajectory(
                    behavior_train, 
                    spike_train, 
                    flight_path_index_train,
                    flight_type_index_train,
                    self.window_size, 
                    self.stride_train
                )
                self.samples_valid = self.split_each_trajectory(
                    behavior_valid, 
                    spike_valid, 
                    flight_path_index_valid,
                    flight_type_index_valid,
                    self.window_size, 
                    self.stride_test
                )
                self.samples_test = self.split_each_trajectory(
                    behavior_test, 
                    spike_test, 
                    flight_path_index_test,
                    flight_type_index_test,
                    self.window_size, 
                    self.stride_test
                )    
            # print('SUM:')
            # print(np.sum(np.array(self.samples_train[2])*1.0 -1.0))
            # print(np.sum(np.array(self.samples_valid[2])*1.0 -1.0))
            # print(np.sum(np.array(self.samples_test[2])*1.0 -1.0))
        else:
            self.samples_train = (np.hstack(behavior_train), np.hstack(spike_train))
            self.samples_valid = (np.hstack(behavior_valid), np.hstack(spike_valid))
            self.samples_test = (np.hstack(behavior_test), np.hstack(spike_test))
            # print(np.hstack(behavior_train).shape, np.hstack(behavior_valid).shape, np.hstack(behavior_test).shape)  # (3, 981) (3, 40) (3, 260)
            # print(np.hstack(spike_train).shape, np.hstack(spike_valid).shape, np.hstack(spike_test).shape)  # (289, 981) (289, 40) (289, 260)

        return (self.samples_train, self.samples_valid, self.samples_test)
    
    @staticmethod
    def split_each_trajectory(
        behavior_list,
        spike_list,
        flight_path_index_list,
        flight_type_index_list,
        window_size=20,
        stride=20,
        pad_on_short=True,
        pad_mode="right",   # "right" (append pad) or "left" (prepend pad)
    ):
        """
        For a single session, split each flight trajectory into multiple samples .
        - behavior_list: list of arrays, each (3, T_k); k: trajectory index
        - spike_list: list of arrays, each (Y, T_k) -- same Y across this session
        - returns:
            samples: a tuple of five lists: b_window_list, s_window_list, mask_list, f_window_path_index_list, f_window_type_index_list
                     each list contains multiple numpy arrays: b_window shape (3, window_size), s_window shape (Y, window_size)
                     mask shape (window_size,) bool (True for real timesteps)
                     and then transform the three lists to three numpy arrays
                - f_window_path_index_list: each element represents the index of individual flight path before the flight path is splitted into multiple windows
                - f_window_type_index_list: each element represents the index of flight cluster type
        Notes:
        - padding values are zeros 
        - if pad_on_short=False, short trajectories are ignored
        """
        assert len(behavior_list) == len(spike_list)

        b_window_list, s_window_list, mask_list, f_window_path_index_list, f_window_type_index_list = [], [], [], [], []

        for b_raw, s_raw, flight_path_index_raw, flight_type_index_raw  in zip(behavior_list, spike_list, flight_path_index_list, flight_type_index_list): 
            TT = b_raw.shape[1]

            if TT >= window_size:
                for start in range(0, TT - window_size +1, stride):
                    b_window = b_raw[:, start: start + window_size].astype(np.float32)
                    s_window = s_raw[:, start: start + window_size].astype(np.float32)
                    mask = np.ones(window_size, dtype=bool)
                    b_window_list.append(b_window)
                    s_window_list.append(s_window)
                    mask_list.append(mask)
                    f_window_path_index_list.append(flight_path_index_raw)
                    f_window_type_index_list.append(flight_type_index_raw)
                # include the last shorter tail
                shorter_tail_length = TT % window_size
                if shorter_tail_length > 0:
                    # create padded arrays
                    if pad_mode == 'right':
                        b_pad = np.full((b_raw.shape[0], window_size), np.nan, dtype=np.float32)
                        s_pad = np.full((s_raw.shape[0], window_size), np.nan, dtype=np.float32)
                        b_pad[:, :shorter_tail_length] = b_raw[:, -shorter_tail_length:]
                        s_pad[:, :shorter_tail_length] = s_raw[:, -shorter_tail_length:]
                        mask = np.zeros(window_size, dtype=bool)
                        mask[:shorter_tail_length] = True
                    elif pad_mode == 'left':
                        b_pad = np.full((b_raw.shape[0], window_size), np.nan, dtype=np.float32)
                        s_pad = np.full((s_raw.shape[0], window_size), np.nan, dtype=np.float32)
                        b_pad[:, window_size-shorter_tail_length:] = b_raw[:, -shorter_tail_length:]
                        s_pad[:, window_size-shorter_tail_length:] = s_raw[:, -shorter_tail_length:]
                        mask = np.zeros(window_size, dtype=bool)
                        mask[window_size-shorter_tail_length:] = True
                    else:
                        raise ValueError("pad_mode must be either 'right' or 'left' ")
                    
                    b_window_list.append(b_pad)
                    s_window_list.append(s_pad)
                    mask_list.append(mask)
                    f_window_path_index_list.append(flight_path_index_raw)
                    f_window_type_index_list.append(flight_type_index_raw)

            else:
                if not pad_on_short:
                    continue
                # create padded arrays
                if pad_mode == 'right':
                    b_pad = np.full((b_raw.shape[0], window_size), np.nan, dtype=np.float32)
                    s_pad = np.full((s_raw.shape[0], window_size), np.nan, dtype=np.float32)
                    b_pad[:, :TT] = b_raw
                    s_pad[:, :TT] = s_raw
                    mask = np.zeros(window_size, dtype=bool)
                    mask[:TT] = True
                elif pad_mode == 'left':
                    b_pad = np.full((b_raw.shape[0], window_size), np.nan, dtype=np.float32)
                    s_pad = np.full((s_raw.shape[0], window_size), np.nan, dtype=np.float32)
                    b_pad[:, window_size-TT:] = b_raw
                    s_pad[:, window_size-TT:] = s_raw
                    mask = np.zeros(window_size, dtype=bool)
                    mask[window_size-TT:] = True
                else:
                    raise ValueError("pad_mode must be either 'right' or 'left' ")
                
                b_window_list.append(b_pad)
                s_window_list.append(s_pad)
                mask_list.append(mask)
                f_window_path_index_list.append(flight_path_index_raw)
                f_window_type_index_list.append(flight_type_index_raw)
                # print(f'b_window_list: {np.array(b_window_list)[:, 1, :]}')
                # print(f'b_window_list.shape: {np.array(b_window_list).shape}')
                # print(f's_window_list: {np.array(s_window_list)[:, 10, :]}')
                # print(f's_window_list.shape: {np.array(s_window_list).shape}')
                # print(f'mask_list: {np.array(mask_list)}')
                # print(f'np.array(mask_list).shape: {np.array(mask_list).shape}')

        return (
            np.array(b_window_list), 
            np.array(s_window_list), 
            np.array(mask_list), 
            np.array(f_window_path_index_list), 
            np.array(f_window_type_index_list),
        ) # shapes: (N, 3, window_size), (N, num_of_neurons, window_size), (N, window_size), (N,), (N,)


def data_prep_for_mlp(all_input, all_label, train_num, valid_num, center_flag=False, use_fourier=False):
    """
    Data preprocessing for MLP.

    Args:
        all_input: np.ndarray, shape (n_samples, n_features)
            Full input data. Positions assumed to be in columns 1 to 3.
        all_label: np.ndarray, shape (n_samples, n_outputs)
            Corresponding labels.
        train_num: int
            Number of training samples.
        valid_num: int
            Number of validation samples.
        use_fourier: bool, default=False
            If True, normalize positions (columns 1–3) to [0, 1] range instead of std.

    Returns:
        Tuple of torch.Tensors:
            train_input, valid_input, test_input,
            train_label, valid_label, test_label
    """

    # Shuffle the data
    np.random.seed(100)
    data_indices = np.random.permutation(all_input.shape[0])
    all_input = all_input[data_indices]
    all_label = all_label[data_indices]

    # Split data
    train_input = all_input[:train_num]
    train_label = all_label[:train_num]
    valid_input = all_input[train_num:train_num + valid_num]
    valid_label = all_label[train_num:train_num + valid_num]
    test_input = all_input[train_num + valid_num:]
    test_label = all_label[train_num + valid_num:]

    # Normalize input
    if use_fourier:
        # Normalize position dims (0:3) to [0, 1]
        pos_min = train_input[:, 0:3].min(axis=0)
        pos_max = train_input[:, 0:3].max(axis=0)
        pos_range = pos_max - pos_min + 1e-8  # avoid div by zero

        def normalize_pos(x):
            x[:, 0:3] = (x[:, 0:3] - pos_min) / pos_range
            return x

        train_input = normalize_pos(train_input)
        valid_input = normalize_pos(valid_input)
        test_input = normalize_pos(test_input)

        # Standardize other dims (0, 3:10)
        keep_dims = list(range(3, all_input.shape[1]))
        train_mean = np.mean(train_input[:, keep_dims], axis=0, keepdims=True) if center_flag else 0.0
        train_std = np.std(train_input[:, keep_dims], axis=0, keepdims=True)
        train_input[:, keep_dims] = (train_input[:, keep_dims] - train_mean) / train_std
        valid_input[:, keep_dims] = (valid_input[:, keep_dims] - train_mean) / train_std
        test_input[:, keep_dims] = (test_input[:, keep_dims] - train_mean) / train_std
    else:
        # Standardize all dimensions
        train_mean = np.mean(train_input, axis=0, keepdims=True) if center_flag else 0.0
        train_std = np.std(train_input, axis=0, keepdims=True)
        train_input = (train_input - train_mean) / train_std
        valid_input = (valid_input - train_mean) / train_std
        test_input = (test_input - train_mean) / train_std

    # Convert to torch tensors
    train_input = torch.from_numpy(train_input).float()
    valid_input = torch.from_numpy(valid_input).float()
    test_input = torch.from_numpy(test_input).float()
    train_label = torch.from_numpy(train_label).float()
    valid_label = torch.from_numpy(valid_label).float()
    test_label = torch.from_numpy(test_label).float()

    return train_input, valid_input, test_input, train_label, valid_label, test_label


class Data_Prep_for_MLP():
    """
    Data preprocessing for MLP.

    Args:
        all_input: np.ndarray, shape (n_samples, n_features)
            Full input data. Positions assumed to be in columns 1 to 3.
        all_label: np.ndarray, shape (n_samples, n_outputs)
            Corresponding labels.
        train_num: int
            Number of training samples.
        valid_num: int
            Number of validation samples.
        use_fourier: bool, default=False
            If True, normalize positions (columns 1–3) to [0, 1] range instead of std.

    Returns:
        Tuple of torch.Tensors:
            train_input, valid_input, test_input,
            train_label, valid_label, test_label
    """

    def __init__(
        self,
        all_input: np.ndarray,
        all_label: np.ndarray,
        train_num: int,
        valid_num: int,
        shuffle_flag: bool = True, 
        center_flag: bool = False, 
        use_fourier: bool = False,
    ):
        super().__init__()
        self.all_input = all_input
        self.all_label = all_label
        self.train_num = train_num
        self.valid_num = valid_num
        self.shuffle_flag = shuffle_flag
        self.center_flag = center_flag
        self.use_fourier = use_fourier
        self.train_mean = None
        self.train_std = None

    def __call__(self):
        # Shuffle the data
        if self.shuffle_flag:
            np.random.seed(100)
            data_indices = np.random.permutation(self.all_input.shape[0])
            all_input = self.all_input[data_indices]
            all_label = self.all_label[data_indices]
        else:
            all_input = np.copy(self.all_input)
            all_label = np.copy(self.all_label)

        # Split data
        train_input = all_input[:self.train_num]
        train_label = all_label[:self.train_num]
        valid_input = all_input[self.train_num : self.train_num + self.valid_num]
        valid_label = all_label[self.train_num : self.train_num + self.valid_num]
        test_input = all_input[self.train_num + self.valid_num : ]
        test_label = all_label[self.train_num + self.valid_num : ]

        # Normalize input
        if self.use_fourier:
            # Normalize position dims (0:3) to [0, 1]
            pos_min = train_input[:, 0:3].min(axis=0)
            pos_max = train_input[:, 0:3].max(axis=0)
            pos_range = pos_max - pos_min + 1e-8  # avoid div by zero

            def normalize_pos(x):
                x[:, 0:3] = (x[:, 0:3] - pos_min) / pos_range
                return x

            train_input = normalize_pos(train_input)
            valid_input = normalize_pos(valid_input)
            test_input = normalize_pos(test_input)

            # Standardize other dims (0, 3:10)
            keep_dims = list(range(3, all_input.shape[1]))
            train_mean = np.mean(train_input[:, keep_dims], axis=0, keepdims=True) if self.center_flag else 0.0
            train_std = np.std(train_input[:, keep_dims], axis=0, keepdims=True)
            train_input[:, keep_dims] = (train_input[:, keep_dims] - train_mean) / train_std
            valid_input[:, keep_dims] = (valid_input[:, keep_dims] - train_mean) / train_std
            test_input[:, keep_dims] = (test_input[:, keep_dims] - train_mean) / train_std
            self.train_mean = train_mean
            self.train_std = train_std

        else:
            # Standardize all dimensions
            train_mean = np.mean(train_input, axis=0, keepdims=True) if self.center_flag else 0.0
            train_std = np.std(train_input, axis=0, keepdims=True)
            train_input = (train_input - train_mean) / train_std
            valid_input = (valid_input - train_mean) / train_std
            test_input = (test_input - train_mean) / train_std
            self.train_mean = train_mean
            self.train_std = train_std

        # Convert to torch tensors
        train_input = torch.from_numpy(train_input).float()
        valid_input = torch.from_numpy(valid_input).float()
        test_input = torch.from_numpy(test_input).float()
        train_label = torch.from_numpy(train_label).float()
        valid_label = torch.from_numpy(valid_label).float()
        test_label = torch.from_numpy(test_label).float()

        return train_input, valid_input, test_input, train_label, valid_label, test_label

    def all_data_prep(self):
        """
        Prepare all data for model prediction
        """
        
        # TODO: self.use_fourier
        all_input_model = (self.all_input - self.train_mean) / self.train_std
        all_input_model = torch.from_numpy(all_input_model).float()
        all_label_model = torch.from_numpy(self.all_label).float()
        
        return all_input_model, all_label_model


class BehaviorSpikeDataset():
    def __init__(
        self, 
        samples,
        # behavior, 
        # spike, 
        behavior_mean=None, 
        behavior_std=None, 
        center_flag=True,
        spike_std=None,
    ):
        """
        samples: a tuple of (behavior, spike)
                    behavior: numpy array of shape (N, num_behavior_channels)
                    spike: numpy array of shape (N, num_neurons)
                or a tuple of (behavior, spike, mask)
        behavior_mean: numpy array of shape (num_feature_channels,) or None
        behavior_std: numpy array of shape (num_feature_channels,) or None
        center: if True, (behavior - mean) / std, else behavior / std
        """
        if len(samples) == 2:
            (behavior, spike) = samples
            mask, flight_path_index, flight_type_index = None, None, None
        elif len(samples) >= 3:
            (behavior, spike, mask, flight_path_index, flight_type_index) = samples
        else:
            raise ValueError(f"'samples' must contain 2 or 3 elements, got {len(samples)}")

        behavior = torch.from_numpy(behavior).float()
        spike = torch.from_numpy(spike).float()
        if mask is not None:
            mask = torch.from_numpy(mask).float()
            flight_path_index = torch.from_numpy(flight_path_index).float()
            flight_type_index = torch.from_numpy(flight_type_index).float()

        if behavior_std is not None:
            behavior_std = torch.from_numpy(behavior_std).float()
            if behavior_mean is not None and center_flag:
                behavior_mean = torch.from_numpy(behavior_mean).float()
                behavior = (behavior - behavior_mean) / behavior_std
            else:
                behavior = behavior / behavior_std

        if spike_std is not None:
            spike_std = torch.from_numpy(spike_std).float()
            spike = spike/(spike_std + 1e-6)

        # set all elements to zero where mask == 0.0
        behavior[mask == 0.0] = 0.0
        spike[mask == 0.0] = 0.0

        self.behavior = behavior
        self.spike = spike
        self.mask = mask
        self.has_mask = mask is not None
        self.flight_path_index = flight_path_index
        self.flight_type_index = flight_type_index

    def __len__(self):
        return len(self.behavior)
    
    def __getitem__(self, idx):
        if self.has_mask:
            DataSample = namedtuple('DataSample',['behavior', 'spike', 'mask', 'flight_path_index', 'flight_type_index'])
            return DataSample(
                behavior=self.behavior[idx],
                spike=self.spike[idx],
                mask=self.mask[idx],
                flight_path_index=self.flight_path_index[idx],
                flight_type_index=self.flight_type_index[idx],
            )
            # return self.behavior[idx], self.spike[idx], self.mask[idx]
        else:
            DataSample = namedtuple('DataSample',['behavior', 'spike'])
            return DataSample(
                behavior=self.behavior[idx],
                spike=self.spike[idx],
            )
            # return self.behavior[idx], self.spike[idx]


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        """
        Adapted from: 
        https://github.com/yongrong-qiu/neuralpredictors/blob/2b420058b2c0c029842ba739829114ddfa0f8b50/neuralpredictors/data/samplers.py#L36

        Samples elements sequentially from a given list of indices, without replacement.
        Arguments:
            indices (list): a list of indices
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def bat_data_loader(
    session_names,
    use_speed_head_dir = False,
    use_IMU_flag = False,  # True
    use_click_flag = False, # True
    use_LFP_flag = False,  # True
    use_ripple_flag = False,
    only_flight_flag = True,  # True
    no_3d_location = False, # False
    spike_count_flag = True,
    bin_width = 0.10, # 0.10, in seconds
    split_flights_for_train = 'manual', # the way to split flight paths for train + valid + test; either 'manual' or 'random'
    type_train_num=1,  # for splitting flight clustering data
    type_valid_num=1,  # for splitting flight clustering data
    special_id_count_thresh=3,  # for splitting flight clustering data
    only_use_special_id_4train_flag=False,
    shuffle_flag=False, # shuffle the training data
    simulate_place_cell_flag=False,
    split_each_trajectory_flag=False, 
    window_size = 20, 
    stride_train = 20, 
    stride_test = 20, 
    batch_size_train=None, # batch_size for training data
    batch_size_test=None,
):
    """
    Dataloaders for multiple sessions of bat data.
    Returns a dictionary of dataloaders, behavior_dim, and a dictionary of spiking unit numbers for each session.
    """
    
    dataloaders_combined = {} # combined dataloaders for train, valid, test and for different sessions
    unit_num_dict = {} # a dictionary of unit/neuron numbers for each session
    # a dictionary of other meta information for each session, including 
    # "r_lim" (boundaries of the room/outdoor areana)
    # "behavior_mean" and "behavior_std" (for data preprocessing before feeding into the model)
    meta_info_dict = {} 
    meta_info_dict['r_lim'] = {}
    meta_info_dict['flight_type_test_indomain'] = {}
    meta_info_dict['flight_type_test_ood'] = {}
    meta_info_dict['behavior_mean'] = {}
    meta_info_dict['behavior_std'] = {}
    meta_info_dict['spike_std'] = {}
    meta_info_dict['spatial_centers_sigma'] = {}

    for session_id, session_name in enumerate(session_names):
        parent_folder = '/mnt/at-globusshare/Users/Yongrong/' + session_name
        su_file_path = '/mnt/at-globusshare/Users/Yongrong/To_transfer/out_Gunits_probe*_' + session_name + '.mat'
        data_prep_for_raw = Data_Prep_for_Raw(
            parent_folder = parent_folder,
            su_file_path = su_file_path,
            use_speed_head_dir = use_speed_head_dir,
            use_IMU_flag = use_IMU_flag,
            use_click_flag = use_click_flag,
            use_LFP_flag = use_LFP_flag,
            use_ripple_flag = use_ripple_flag,
            only_flight_flag = only_flight_flag,
            no_3d_location = no_3d_location,
            spike_count_flag = spike_count_flag,
            bin_width = bin_width, 
            split_flights_for_train = split_flights_for_train, 
            type_train_num = type_train_num,
            type_valid_num = type_valid_num,
            special_id_count_thresh = special_id_count_thresh,
            only_use_special_id_4train_flag = only_use_special_id_4train_flag,
            simulate_place_cell_flag = simulate_place_cell_flag,
            split_each_trajectory_flag = split_each_trajectory_flag,
            window_size = window_size, 
            stride_train = stride_train, 
            stride_test = stride_test, 
        )
        
        all_behavior = data_prep_for_raw.get_all_behavior()
        meta_info_dict['r_lim'][session_name] = data_prep_for_raw.r_lim
        all_spike = data_prep_for_raw.get_all_spike()
        spike_ratio = data_prep_for_raw.spike_ratio
        # print(f'all_behavior.shape: {all_behavior.shape}') # all_behavior.shape: (3, 2820), 3 represents behavior features
        # print(f'all_spike.shape: {all_spike.shape}') # all_spike.shape: (95, 2820), 95 represents number of units
        # print(f'spike_ratio.shape: {spike_ratio.shape}') # spike_ratio.shape: (95,)
        
        # flight cluster
        all_behavior_clustered, all_behavior_type, all_spike_clustered = data_prep_for_raw.get_flight_cluster()
        # split the flight-cluster data
        (samples_train, samples_valid, samples_test) = data_prep_for_raw.split_flight_cluster()
        meta_info_dict['flight_type_test_indomain'][session_name] = data_prep_for_raw.flight_type_test_indomain
        meta_info_dict['flight_type_test_ood'][session_name] = data_prep_for_raw.flight_type_test_ood

        if len(samples_train) == 2:
            (behavior_train, spike_train) = samples_train
            (behavior_valid, spike_valid) = samples_valid
            (behavior_test, spike_test) = samples_test
            # print(behavior_train.shape, behavior_valid.shape, behavior_test.shape)  # (3, 843) (3, 229) (3, 1732), 3 represents behavior features
            # print(spike_train.shape, spike_valid.shape, spike_test.shape)  # (95, 843) (95, 229) (95, 1732), 95 represents number of units
            mask_train = None

            behavior_dim = behavior_train.shape[0]
            unit_num_dict[session_name] = spike_train.shape[0]

        elif len(samples_train) >= 3: # for modeling temporal dependencies
            # shapes: (N, 3, window_size), (N, num_of_neurons, window_size), (N, window_size), (N,), (N,)
            (behavior_train, spike_train, mask_train, flight_path_index_train, flight_type_index_train) = samples_train 
            (behavior_valid, spike_valid, mask_valid, flight_path_index_valid, flight_type_index_valid) = samples_valid
            (behavior_test, spike_test, mask_test, flight_path_index_test, flight_type_index_test) = samples_test

            behavior_dim = behavior_train.shape[1]
            unit_num_dict[session_name] = spike_train.shape[1]

        else:
            raise ValueError(f"'samples_train' must contain 2 or 3 elements, got {len(samples_train)}")

        # pytorch dataset and dataloaders
        dataloaders = {}

        # train
        if mask_train is None:
            data_input = np.copy(behavior_train.T)
            data_output = np.copy(spike_train.T)
            behavior_mean = np.mean(data_input, axis=0, keepdims=True)
            behavior_std = np.std(data_input, axis=0, keepdims=True)
            spike_std = np.std(data_output, axis=0, keepdims=True)
        else:
            data_input = np.transpose(behavior_train, (0, 2, 1))
            data_output = np.transpose(spike_train, (0, 2, 1))
            data_mask = np.copy(mask_train) # 1.0 for real timesteps
            data_flight_path_index = np.copy(flight_path_index_train)
            data_flight_type_index = np.copy(flight_type_index_train)
            behavior_mean = np.nanmean(data_input, axis=(0,1), keepdims=True)
            behavior_std = np.nanstd(data_input, axis=(0,1), keepdims=True)
            # if use_speed_head_dir: # somehow this does not help
            #     # when adding speed (the forth channel) and head direction (unit, the fifth to seventh channels)
            #     # we scale the speed to std=1.0, we do not change head direction
            #     behavior_mean[:, :, 3:7] = 0.0
            #     behavior_std[:, :, 4:7] = 1.0
            spike_std = np.nanstd(data_output, axis=(0,1), keepdims=True)
            data_input = np.nan_to_num(data_input) # NaN to 0.0, shape: (N, window_size, num_behavior_channels)
            data_output = np.nan_to_num(data_output) # NaN to 0.0, shape: (N, window_size, num_neurons)
            centers, sigma = compute_rbf_basis_from_training(
                (data_input[data_mask == 1.0] - behavior_mean.squeeze(axis=0))/behavior_std.squeeze(axis=0), 
                K=32,
            )
        meta_info_dict['behavior_mean'][session_name] = behavior_mean
        meta_info_dict['behavior_std'][session_name] = behavior_std
        meta_info_dict['spike_std'][session_name] = spike_std
        meta_info_dict['spatial_centers_sigma'][session_name] = (centers, sigma)
        ind = list(range(data_input.shape[0]))
        dat = BehaviorSpikeDataset(
            samples = (data_input, data_output) if mask_train is None else (data_input, data_output, data_mask, data_flight_path_index, data_flight_type_index),
            behavior_mean = behavior_mean,
            behavior_std = behavior_std,
            # spike_std = spike_std,
        )
        dataloaders['train'] = DataLoader(
            dat,
            # TODO: use random sampler, but still keep the order within each trajectory
            sampler = SubsetRandomSampler(ind) if shuffle_flag else SubsetSequentialSampler(ind),
            batch_size = data_input.shape[0] if batch_size_train is None else batch_size_train,
        )

        # validation
        if mask_train is None:
            data_input = np.copy(behavior_valid.T)
            data_output = np.copy(spike_valid.T)
        else:
            data_input = np.transpose(behavior_valid, (0, 2, 1))
            data_output = np.transpose(spike_valid, (0, 2, 1))
            data_mask = np.copy(mask_valid)
            data_flight_path_index = np.copy(flight_path_index_valid)
            data_flight_type_index = np.copy(flight_type_index_valid)
            data_input = np.nan_to_num(data_input) 
            data_output = np.nan_to_num(data_output) 
        ind = list(range(data_input.shape[0]))
        dat = BehaviorSpikeDataset(
            samples = (data_input, data_output) if mask_train is None else (data_input, data_output, data_mask, data_flight_path_index, data_flight_type_index),
            behavior_mean = behavior_mean,
            behavior_std = behavior_std,
            # spike_std = spike_std,
        )
        dataloaders['valid'] = DataLoader(
            dat,
            sampler = SubsetSequentialSampler(ind),
            batch_size = data_input.shape[0] if batch_size_train is None else batch_size_train,
        )

        # test
        if mask_train is None:
            data_input = np.copy(behavior_test.T)
            data_output = np.copy(spike_test.T)
        else:
            data_input = np.transpose(behavior_test, (0, 2, 1))
            data_output = np.transpose(spike_test, (0, 2, 1))
            data_mask = np.copy(mask_test)
            data_flight_path_index = np.copy(flight_path_index_test)
            data_flight_type_index = np.copy(flight_type_index_test)
            data_input = np.nan_to_num(data_input) 
            data_output = np.nan_to_num(data_output) 
        ind = list(range(data_input.shape[0]))
        dat = BehaviorSpikeDataset(
            samples = (data_input, data_output) if mask_train is None else (data_input, data_output, data_mask, data_flight_path_index, data_flight_type_index),
            behavior_mean = behavior_mean,
            behavior_std = behavior_std,
            # spike_std = spike_std,
        )
        dataloaders['test'] = DataLoader(
            dat,
            sampler = SubsetSequentialSampler(ind),
            batch_size = data_input.shape[0] if batch_size_test is None else batch_size_test,
        )

        for k,v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][session_name] = v

    return dataloaders_combined, behavior_dim, unit_num_dict, meta_info_dict


def compute_rbf_basis_from_training(train_trajectories, K, random_state=0):
    """
    Parameters
    ----------
    train_trajectories : numpy array with a shape of (T, 3), concatenating all flight trajectories in the training dataset
    K : int
        Number of RBF centers.
    random_state : int
        Random seed for KMeans.

    Returns
    -------
    centers : (K, 3) numpy array
        RBF centers.
    sigma : float
        Global RBF width.
    """

    # # Concatenate all training trajectories
    # X = np.concatenate(train_trajectories, axis=0)  # (sum_i T_i, 3)
    # KMeans to get centers
    kmeans = KMeans(n_clusters=K, random_state=random_state)
    kmeans.fit(train_trajectories)
    centers = kmeans.cluster_centers_  # (K, 3), numpy array
    # Pairwise distances between centers
    dists = cdist(centers, centers)  # (K, K)
    # Exclude zero distances (diagonal)
    nonzero_dists = dists[dists > 0]
    # Robust estimate of spatial scale
    sigma = np.median(nonzero_dists)

    return centers, float(sigma)


class PlaceCellSimulator3D:

    def __init__(self, bin_size=0.3, smoothing_sigma=None):
        """
        bin_size : spatial bin size (in meter or other unit)
        smoothing_sigma : Gaussian smoothing sigma (in bins)
        """
        self.bin_size = bin_size
        self.smoothing_sigma = smoothing_sigma

    def _compute_bins(self, behavior_train):
        """Compute spatial bin edges."""

        all_pos = np.concatenate([b.T for b in behavior_train], axis=0)
        mins = all_pos.min(axis=0)
        maxs = all_pos.max(axis=0)

        self.edges = [
            np.arange(mins[i], maxs[i] + self.bin_size, self.bin_size)
            for i in range(3)
        ]
        self.shape = (
            len(self.edges[0]) - 1,
            len(self.edges[1]) - 1,
            len(self.edges[2]) - 1,
        )

    def fit(self, behavior_train, spike_train, dt):
        """
        Compute rate maps.

        behavior_train : list of arrays (3,T)
        spike_train : list of arrays (N,T)
        dt : sampling time
        """

        self._compute_bins(behavior_train)
        N = spike_train[0].shape[0]

        occupancy = np.zeros(self.shape)
        spike_counts = np.zeros((N,) + self.shape)

        for beh, spk in zip(behavior_train, spike_train):
            pos = beh.T
            # -------- occupancy using histogramdd --------
            occ, _ = np.histogramdd(pos, bins=self.edges)
            occupancy += occ * dt
            # -------- spike counts --------
            for n in range(N):
                spike_hist, _ = np.histogramdd(
                    pos,
                    bins=self.edges,
                    weights=spk[n]
                )
                spike_counts[n] += spike_hist

        rate_map = spike_counts / (occupancy + 1e-8)
        # -------- optional Gaussian smoothing --------
        if self.smoothing_sigma is not None:
            for n in range(N):
                rate_map[n] = gaussian_filter(
                    rate_map[n],
                    sigma=self.smoothing_sigma
                )
        self.rate_map = rate_map

        return self.rate_map

    def simulate(self, behavior_test, dt, poisson=False):
        """
        Generate spikes for new trajectories.

        behavior_test : list of arrays (3,T)

        returns
        -------
        spike_test : list of arrays (N,T)
        """

        N = self.rate_map.shape[0]
        spike_test = []

        for beh in behavior_test:
            pos = beh.T
            T = pos.shape[0]
            ix = np.digitize(pos[:, 0], self.edges[0]) - 1
            iy = np.digitize(pos[:, 1], self.edges[1]) - 1
            iz = np.digitize(pos[:, 2], self.edges[2]) - 1

            spikes = np.zeros((N, T))
            valid = (
                (ix >= 0) & (ix < self.shape[0]) &
                (iy >= 0) & (iy < self.shape[1]) &
                (iz >= 0) & (iz < self.shape[2])
            )

            for t in np.where(valid)[0]:
                rate = self.rate_map[:, ix[t], iy[t], iz[t]]
                lam = rate * dt
                if poisson:
                    spikes[:, t] = np.random.poisson(lam)
                else:
                    spikes[:, t] = lam

            spike_test.append(spikes)

        return spike_test
    

def plot_2d_trajectory_4_simulation(
    behavior_train,
    spike_train,
    behavior_train_simulation,
    spike_train_simulation,
    behavior_test,
    spike_test,
    behavior_test_simulation,
    spike_test_simulation,
    units,
    loc_bin_width,
    dt,
):
    """
    Plot 2D trajectory for the simulated 'pure place cell'
    behavior_train : list of arrays (3,T)
    spike_train : list of arrays (N,T)
    """
    all_pos = np.concatenate([b.T for b in behavior_train], axis=0)
    loc_mins = all_pos.min(axis=0) # shape: (3, )
    loc_maxs = all_pos.max(axis=0) # shape: (3, )
    all_spike = np.concatenate([b.T for b in spike_train], axis=0)
    spike_mins = all_spike.min(axis=0) # shape: (N, )
    spike_maxs = all_spike.max(axis=0) # shape: (N, )
    
    behavior_lists = [behavior_train, behavior_train_simulation, behavior_test, behavior_test_simulation]
    spike_lists = [spike_train, spike_train_simulation, spike_test, spike_test_simulation]
    data_label_list = ['Train', 'Train_sim', 'Test', 'Test_sim']

    ncols = len(units)
    nrows = len(behavior_lists)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*1.5))
    for ii in range(nrows): # training, test, test_simulation
        for jj, unit in enumerate(units): # each neuron/unit
            loc_xx = np.concatenate([b.T for b in behavior_lists[ii]], axis=0)[:, 0]
            loc_yy = np.concatenate([b.T for b in behavior_lists[ii]], axis=0)[:, 1]
            spike_count = np.concatenate([s.T for s in spike_lists[ii]], axis=0)[:, unit]
            # Make line segments between successive points
            points = np.array([loc_xx, loc_yy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Normalize spike counts for colormap
            norm = plt.Normalize(spike_mins[unit], spike_maxs[unit])
            # Create a line collection with color mapping
            lc = LineCollection(segments, cmap='hot', norm=norm)
            lc.set_array(spike_count)
            lc.set_linewidth(1)
            # plot
            line = axs[ii,jj].add_collection(lc)
            if ii == 0:
                axs[ii,jj].set_title(f"Unit: {unit}")
            axs[ii,jj].set_xlim(loc_mins[0], loc_maxs[0])
            axs[ii,jj].set_ylim(loc_mins[1], loc_maxs[1])
            if ii == nrows-1:
                axs[ii,jj].set_xlabel("X")
            if jj == 0:
                axs[ii,jj].set_ylabel(data_label_list[ii])
            plt.colorbar(line, ax=axs[ii,jj], label="Spike count")
            plt.tight_layout()

    from ..utility.utils import generate_2d_rate_map
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*1.5))
    for jj, unit in enumerate(units): # each neuron/unit
        for ii in range(nrows): # training, test, test_simulation
            loc_xx = np.concatenate([b.T for b in behavior_lists[ii]], axis=0)[:, 0]
            loc_yy = np.concatenate([b.T for b in behavior_lists[ii]], axis=0)[:, 1]
            spike_count = np.concatenate([s.T for s in spike_lists[ii]], axis=0)[:, unit]
            #
            x_edges, y_edges, rate_map, smoothed_rate_map = generate_2d_rate_map(
                loc_xx=loc_xx,
                loc_yy=loc_yy,
                unit_spike_count=spike_count,
                loc_xx_lim=(loc_mins[0], loc_maxs[0]),
                loc_yy_lim=(loc_mins[1], loc_maxs[1]),
                loc_bin_width=loc_bin_width,
                dt=dt,
            )
            if ii==0: # Be carefule: for each neuron, we use the limits of the training data
                vmin, vmax = np.min(rate_map), np.max(rate_map)
            im = axs[ii,jj].imshow(rate_map.T, origin='lower', # vmin=vmin, vmax=vmax,
                        extent=[loc_mins[0], loc_maxs[0], loc_mins[1], loc_maxs[1]],
                        aspect='auto', cmap='hot_r')
            if ii == 0:
                axs[ii,jj].set_title(f"Unit: {unit}")
            # axs[ii,jj].set_xlim(loc_mins[0], loc_maxs[0])
            # axs[ii,jj].set_ylim(loc_mins[1], loc_maxs[1])
            if ii == nrows-1:
                axs[ii,jj].set_xlabel("X")
            if jj == 0:
                axs[ii,jj].set_ylabel(data_label_list[ii])
            plt.colorbar(im, ax=axs[ii,jj], label="Spikes/s")
            plt.tight_layout()

    return None
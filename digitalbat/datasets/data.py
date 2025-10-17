import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.io import loadmat
import glob
import h5py
from scipy.stats import binned_statistic

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
        use_IMU_flag,
        use_click_flag,
        use_LFP_flag,
        use_ripple_flag,
        only_flight_flag,
        no_3d_location=False, # default: we use 3d locations
        spike_count_flag=False, # default: we use spiking event, not spike counts 
        bin_width=0.010, # 10 ms = 0.01 seconds, so the new sampling frequency is 100 Hz
        flight_cluster_alpha=0.8, # alpha for flight clustering
        type_train_num=1,  # for splitting flight clustering data
        type_valid_num=1,  # for splitting flight clustering data
        special_id_count_thresh=3,  # for splitting flight clustering data
        # After splitting fight-clustering data into train, validation and test, we may want to use each flight trajectory as a data sample.
        # With such data, we can build a network (such as Transformer) to model the temporal dependency of animal behavior (and neural activity) during each flight 
        trajectory_per_sample_flag=False, 
    ):
        super().__init__()
        self.parent_folder = parent_folder
        self.su_file_path = glob.glob(su_file_path)  # a list of strings
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
        self.trajectory_per_sample_flag = trajectory_per_sample_flag

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
        self.all_behavior_cluster = None  # a list of arrays, each array contains behavior for each trajectory
        self.all_behavior_type = None  # a list of trajectory clustering types
        self.all_spike_cluster = None  # a list of arrays, each array contains neuronal activity for each trajectory

        # after flight clustering, split the trajectories for training, evaluation and test
        self.behavior_train = None
        self.behavior_valid = None
        self.behavior_test = None
        self.spike_train = None
        self.spike_valid = None
        self.spike_test = None

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
        bhv_v_abs = bhv['bflying'][:,0] # 'v_abs'
        # bhv_v_abs[np.where(bhv_v_abs > bhv_thresh)] = bhv_thresh
        bhv_r = bhv['r']
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
        # flight status
        temp = interpolate_signal_with_nans(bhv_t, bhv_v_abs, self.all_time)
        # print (f'temp.shape: {temp.shape}')
        all_behavior.append(temp)

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
            self.flight_ind = np.where(all_behavior[3]>0.0)[0]
            self.all_time_flight = self.all_time[self.flight_ind]
            all_behavior = all_behavior[:, self.flight_ind]
            # all_spike = all_spike[:, flight_ind]
            # remove flying dimension from the all_behavior
            all_behavior = np.delete(all_behavior, 3, axis=0)
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
        all_behavior_cluster = []
        all_spike_cluster = []
        all_behavior_type = []
        for tempid in unique_ids:
            start_time = bhv_t[self.flight_cluster_results['start_frame'][np.where(self.flight_cluster_results['id']==tempid)[0]]]
            stop_time = bhv_t[self.flight_cluster_results['stop_frame'][np.where(self.flight_cluster_results['id']==tempid)[0]]]
            for ii in range(len(start_time)):
                tempind = np.where((self.all_time_flight>=start_time[ii]) & (self.all_time_flight<=stop_time[ii]))[0]
                if len(tempind): # exclude the flight behavior beyond all_time_flight, e.g., at the very beginning of the session
                    all_behavior_cluster.append(self.all_behavior[:, tempind])
                    all_behavior_type.append(tempid)
                    all_spike_cluster.append(self.all_spike[:, tempind])
    
        # # shuffle all flights
        # np.random.seed(100)
        # np.random.shuffle(all_behavior_cluster)
        # np.random.shuffle(all_spike_cluster)

        self.all_behavior_cluster = all_behavior_cluster
        self.all_behavior_type = all_behavior_type
        self.all_spike_cluster = all_spike_cluster

        self._flight_cluster_ready = True
        
        return self.all_behavior_cluster, self.all_behavior_type, self.all_spike_cluster
    
    def split_flight_cluster(self):
        """
        After performing flight clustering, we split trajectories for training, validation and test
        """
        if not self._flight_cluster_ready:
            raise RuntimeError('You must run get_flight_cluster() before running split_flight_cluster().')

        # After flight clustering, some types have multiple repeats while others have only one repeat
        # There are several ways to use the data for modeling.
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

        behavior_train, behavior_valid, behavior_test = [], [], []
        spike_train, spike_valid, spike_test = [], [], []
        for ii in range(len(self.all_behavior_type)):
            if self.all_behavior_type[ii] not in special_ids:
                behavior_train.append(self.all_behavior_cluster[ii])
                spike_train.append(self.all_spike_cluster[ii])
        for unique_id in special_ids:
            tempinds = np.where(np.array(self.all_behavior_type)==unique_id)[0]
            for jj in range(len(tempinds)):
                if jj < self.type_train_num:
                    behavior_train.append(self.all_behavior_cluster[tempinds[jj]])
                    spike_train.append(self.all_spike_cluster[tempinds[jj]])
                elif jj < self.type_train_num + self.type_valid_num:
                    behavior_valid.append(self.all_behavior_cluster[tempinds[jj]])
                    spike_valid.append(self.all_spike_cluster[tempinds[jj]])
                else:
                    behavior_test.append(self.all_behavior_cluster[tempinds[jj]])
                    spike_test.append(self.all_spike_cluster[tempinds[jj]])

        # print(len(behavior_train), len(behavior_valid), len(behavior_test))  # 22 2 13
        # print(len(spike_train), len(spike_valid), len(spike_test))  # 22 2 13
        # shuffle the training trajectories
        np.random.seed(100)
        np.random.shuffle(behavior_train)
        np.random.seed(100)
        np.random.shuffle(spike_train)
        if self.trajectory_per_sample_flag:
            self.behavior_train = np.hstack(behavior_train)
            self.behavior_valid = np.hstack(behavior_valid)
            self.behavior_test = np.hstack(behavior_test)
            self.spike_train = np.hstack(spike_train)
            self.spike_valid = np.hstack(spike_valid)
            self.spike_test = np.hstack(spike_test)
            print(f'Maximum flight length: {np.max([temp.shape[1] for temp in self.all_behavior_cluster])}')
            pass
        else:
            self.behavior_train = np.hstack(behavior_train)
            self.behavior_valid = np.hstack(behavior_valid)
            self.behavior_test = np.hstack(behavior_test)
            self.spike_train = np.hstack(spike_train)
            self.spike_valid = np.hstack(spike_valid)
            self.spike_test = np.hstack(spike_test)
            # print(behavior_train.shape, behavior_valid.shape, behavior_test.shape)  # (3, 981) (3, 40) (3, 260)
            # print(spike_train.shape, spike_valid.shape, spike_test.shape)  # (289, 981) (289, 40) (289, 260)

        return self.behavior_train, self.behavior_valid, self.behavior_test, self.spike_train, self.spike_valid, self.spike_test


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
        behavior, 
        spike, 
        behavior_mean=None, 
        behavior_std=None, 
        center_flag=True,
        spike_std=None,
    ):
        """
        behavior: numpy array of shape (N, num_feature_channels)
        spike: numpy array of shape (N, num_neurons)
        behavior_mean: numpy array of shape (num_feature_channels,) or None
        behavior_std: numpy array of shape (num_feature_channels,) or None
        center: if True, (behavior - mean) / std, else behavior / std
        """
        behavior = torch.from_numpy(behavior).float()
        spike = torch.from_numpy(spike).float()

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

        self.behavior = behavior
        self.spike = spike

    def __len__(self):
        return len(self.behavior)
    
    def __getitem__(self, idx):
        return self.behavior[idx], self.spike[idx]


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
    use_IMU_flag = False,  # True
    use_click_flag = False, # True
    use_LFP_flag = False,  # True
    use_ripple_flag = False,
    only_flight_flag = True,  # True
    no_3d_location = False, # False
    spike_count_flag = True,
    bin_width = 0.10, # 0.10, in seconds
    type_train_num=1,  # for splitting flight clustering data
    type_valid_num=1,  # for splitting flight clustering data
    special_id_count_thresh=3,  # for splitting flight clustering data
    shuffle_flag=False, # shuffle the training data
    trajectory_per_sample_flag=False, 
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
    meta_info_dict['behavior_mean'] = {}
    meta_info_dict['behavior_std'] = {}
    meta_info_dict['spike_std'] = {}

    for session_id, session_name in enumerate(session_names):
        parent_folder = '/mnt/at-globusshare/Users/Yongrong/' + session_name
        su_file_path = '/mnt/at-globusshare/Users/Yongrong/To_transfer/out_Gunits_probe*_' + session_name + '.mat'
        data_prep_for_raw = Data_Prep_for_Raw(
            parent_folder = parent_folder,
            su_file_path = su_file_path,
            use_IMU_flag = use_IMU_flag,
            use_click_flag = use_click_flag,
            use_LFP_flag = use_LFP_flag,
            use_ripple_flag = use_ripple_flag,
            only_flight_flag = only_flight_flag,
            no_3d_location = no_3d_location,
            spike_count_flag = spike_count_flag,
            bin_width = bin_width, 
            type_train_num = type_train_num,
            type_valid_num = type_valid_num,
            special_id_count_thresh = special_id_count_thresh,
            trajectory_per_sample_flag = trajectory_per_sample_flag,
        )
        
        all_behavior = data_prep_for_raw.get_all_behavior()
        meta_info_dict['r_lim'][session_name] = data_prep_for_raw.r_lim
        all_spike = data_prep_for_raw.get_all_spike()
        spike_ratio = data_prep_for_raw.spike_ratio
        # print(f'all_behavior.shape: {all_behavior.shape}') # all_behavior.shape: (3, 2820), 3 represents behavior features
        # print(f'all_spike.shape: {all_spike.shape}') # all_spike.shape: (95, 2820), 95 represents number of units
        # print(f'spike_ratio.shape: {spike_ratio.shape}') # spike_ratio.shape: (95,)
        
        # flight cluster
        all_behavior_cluster, all_behavior_type, all_spike_cluster = data_prep_for_raw.get_flight_cluster()
        # split the flight-cluster data
        behavior_train, behavior_valid, behavior_test, spike_train, spike_valid, spike_test = data_prep_for_raw.split_flight_cluster()
        # print(behavior_train.shape, behavior_valid.shape, behavior_test.shape)  # (3, 843) (3, 229) (3, 1732), 3 represents behavior features
        # print(spike_train.shape, spike_valid.shape, spike_test.shape)  # (95, 843) (95, 229) (95, 1732), 95 represents number of units

        behavior_dim = behavior_train.shape[0]
        unit_num_dict[session_name] = spike_train.shape[0]

        # pytorch dataset and dataloaders
        dataloaders = {}
        # train
        data_input = np.copy(behavior_train.T)
        data_output = np.copy(spike_train.T)
        behavior_mean = np.mean(data_input, axis=0, keepdims=True)
        behavior_std = np.std(data_input, axis=0, keepdims=True)
        spike_std = np.std(data_output, axis=0, keepdims=True)
        meta_info_dict['behavior_mean'][session_name] = behavior_mean
        meta_info_dict['behavior_std'][session_name] = behavior_std
        meta_info_dict['spike_std'][session_name] = spike_std
        ind = list(range(data_input.shape[0]))
        dat = BehaviorSpikeDataset(
            behavior = data_input,
            spike = data_output,
            behavior_mean = behavior_mean,
            behavior_std = behavior_std,
            # spike_std = spike_std,
        )
        dataloaders['train'] = DataLoader(
            dat,
            # TODO: use random sampler, but still keep the order within each trajectory
            sampler = SubsetRandomSampler(ind) if shuffle_flag else SubsetSequentialSampler(ind),
            batch_size = data_input.shape[0],
        )
        # validation
        data_input = np.copy(behavior_valid.T)
        data_output = np.copy(spike_valid.T)
        ind = list(range(data_input.shape[0]))
        dat = BehaviorSpikeDataset(
            behavior = data_input,
            spike = data_output,
            behavior_mean = behavior_mean,
            behavior_std = behavior_std,
            # spike_std = spike_std,
        )
        dataloaders['valid'] = DataLoader(
            dat,
            sampler = SubsetSequentialSampler(ind),
            batch_size = data_input.shape[0],
        )
        # test
        data_input = np.copy(behavior_test.T)
        data_output = np.copy(spike_test.T)
        ind = list(range(data_input.shape[0]))
        dat = BehaviorSpikeDataset(
            behavior = data_input,
            spike = data_output,
            behavior_mean = behavior_mean,
            behavior_std = behavior_std,
            # spike_std = spike_std,
        )
        dataloaders['test'] = DataLoader(
            dat,
            sampler = SubsetSequentialSampler(ind),
            batch_size = data_input.shape[0],
        )

        for k,v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][session_name] = v

    return dataloaders_combined, behavior_dim, unit_num_dict, meta_info_dict



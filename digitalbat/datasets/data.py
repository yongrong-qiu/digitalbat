import numpy as np
import torch
from scipy.io import loadmat
import glob
import h5py
from scipy.stats import binned_statistic

from ..utility.utils import bandpass_filter
from ..utility.utils import interpolate_signal_with_nans


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
        bin_width=0.010, # 10 ms = 0.01 seconds, so the new sampling frequency is 100 Hz
    ):
        super().__init__()
        self.parent_folder = parent_folder
        self.su_file_path = su_file_path
        self.use_IMU_flag = use_IMU_flag
        self.use_click_flag = use_click_flag
        self.use_LFP_flag = use_LFP_flag,
        self.use_ripple_flag = use_ripple_flag
        self.only_flight_flag = only_flight_flag

        # data resampling
        self.bin_width = bin_width
        self.global_t_start = 0.0 + bin_width / 2
        self.global_t_end = None
        self.all_time = None

        self.all_behavior = None
        self.all_spike = None
        self.flight_ind = None
        self.spike_ratio = None # spike ratio for each unit after resampling

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
        bhv_t = bhv['t'][0,:] # np.arange(len(bhv_v_abs))/bhv['Fs'][0]

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
        self.all_time =np.arange(self.global_t_start, self.global_t_end, self.bin_width)

        # behavior resampling
        all_behavior = []
        for ii in range(bhv_r.shape[1]):
            temp = interpolate_signal_with_nans(bhv_t, bhv_r[:,ii], self.all_time)
            # print (f'temp.shape: {temp.shape}')
            all_behavior.append(temp)

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

        if self.use_LFP_flag:
            # lfp_filter, beta_band: Compute mean power in each bin (vectorized); and align
            binned_power, _, _ = binned_statistic(
                lfp_t_ds,                   # Time stamps
                lfp_filter ** 2,            # Power
                statistic='mean',
                bins=np.append(self.all_time, self.all_time[-1] + self.bin_width) - self.bin_width/2,        # Bin edges
                range=(self.all_time[0] - self.bin_width / 2, self.all_time[-1] + self.bin_width / 2)
            )
            all_behavior.append(binned_power)
        all_behavior = np.array(all_behavior)
        # print (f'all_behavior.shape: {all_behavior.shape}') # all_behavior.shape: (11, 524859)

        # only use the flight data
        if self.only_flight_flag:
            self.flight_ind = np.where(all_behavior[3]>0.0)[0]
            # print (f'Ratio of flight: {len(flight_ind)/all_behavior.shape[1]}')
            all_behavior = all_behavior[:, self.flight_ind]
            # all_spike = all_spike[:, flight_ind]
            # remove flying dimension from the all_behavior
            all_behavior = np.delete(all_behavior, 3, axis=0)
            # print (f'all_behavior.shape: {all_behavior.shape}') 

        self.all_behavior = all_behavior
        
        return self.all_behavior

    def get_all_spike(self):
        """
        Get the single-unit spikes 
        """
        su_good = loadmat(self.su_file_path)

        su_good1_cluster_id = np.array([x[0][0,0] for x in su_good['out_good_units1']['cluster_id']])
        # print (f'len(su_good1_cluster_id): {len(su_good1_cluster_id)}, su_good1_cluster_id: {su_good1_cluster_id}')
        su_good1_time = [x[0][:,0]/1e6 for x in su_good['out_good_units1']['spikeTimes_usec']]
        # print (f'len(su_good1_time): {len(su_good1_time)}, su_good1_time[3].shape: {su_good1_time[3].shape}')

        su_good2_cluster_id = np.array([x[0][0,0] for x in su_good['out_good_units2']['cluster_id']])
        # print (f'len(su_good2_cluster_id): {len(su_good2_cluster_id)}, su_good2_cluster_id: {su_good2_cluster_id}')
        su_good2_time = [x[0][:,0]/1e6 for x in su_good['out_good_units2']['spikeTimes_usec']]
        # print (f'len(su_good2_time): {len(su_good2_time)}, su_good2_time[3].shape: {su_good2_time[3].shape}')

        su_good3_cluster_id = np.array([x[0][0,0] for x in su_good['out_good_units3']['cluster_id']])
        # print (f'len(su_good3_cluster_id): {len(su_good3_cluster_id)}, su_good3_cluster_id: {su_good3_cluster_id}')
        su_good3_time = [x[0][:,0]/1e6 for x in su_good['out_good_units3']['spikeTimes_usec']]
        # print (f'len(su_good3_time): {len(su_good3_time)}, su_good3_time[3].shape: {su_good3_time[3].shape}')

        # merge units
        mergesFile = glob.glob(self.parent_folder + '/merges_Dataset_*.mat')[0]
        merges = loadmat(mergesFile)
        # here represents three probes; the first probe does not have any units need to be merged
        # for the second probe, the cluster_ids  of 170 and 176 needs to be merged
        # for the third probe, the cluster_ids of 47, 66, 74 need to be merged
        # we use refractory period of 1 ms to merge two very closed spikes 

        refractory = 0.001 # refractory period of 1 ms
        probe_num = 3 # three probes
        su_goods_cluster_id = [su_good1_cluster_id, su_good2_cluster_id, su_good3_cluster_id]
        su_goods_time = [su_good1_time, su_good2_time, su_good3_time]
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

        # bin spike times
        t_start = self.global_t_start - self.bin_width / 2
        t_end = np.ceil(self.global_t_end / self.bin_width) * self.bin_width
        bin_edges = np.arange(t_start, t_end + self.bin_width, self.bin_width)
        bin_centers = bin_edges[:-1] + self.bin_width / 2

        # su_goods_time = [su_good1_time, su_good2_time, su_good3_time]
        all_spike = []
        for su_good_time in su_goods_time:
            for ii in range(len(su_good_time)):
                # Bin the spike times
                spike_counts, _ = np.histogram(su_good_time[ii], bins=bin_edges)
                all_spike.append(spike_counts)
        all_spike = np.array(all_spike)
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

        return all_spike


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
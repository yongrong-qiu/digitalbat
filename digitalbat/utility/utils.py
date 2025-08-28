import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from numpy.typing import ArrayLike
from typing import Tuple, Union


def bandpass_filter(signal, fs, band):
    """Bandpass filter a signal."""
    nyq = fs / 2
    b, a = butter(3, [band[0] / nyq, band[1] / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_band_power_per_bin(lfp, fs, band, bin_size_sec):
    """
    Bandpass filter the LFP, compute the envelope using the Hilbert transform,
    and bin the power into fixed-length windows.
    """
    # Step 1: Filter the signal to the desired band
    filtered = bandpass_filter(lfp, fs, band)
    
    # Step 2: Compute the envelope using Hilbert transform
    envelope = np.abs(hilbert(filtered))
    
    # Step 3: Bin the envelope
    bin_size = int(bin_size_sec * fs)
    n_bins = len(envelope) // bin_size
    binned_power = np.array([
        np.mean(envelope[i * bin_size:(i + 1) * bin_size]**2)
        for i in range(n_bins)
    ])
    
    return binned_power

def corr(
    y1: ArrayLike, y2: ArrayLike, axis: Union[None, int, Tuple[int]] = -1, eps: int = 1e-8, **kwargs
) -> np.ndarray:
    """
    Copied from:
    https://github.com/yongrong-qiu/neuralpredictors/blob/2b420058b2c0c029842ba739829114ddfa0f8b50/neuralpredictors/measures/np_functions.py#L13
    
    Compute the correlation between two NumPy arrays along the specified dimension(s).

    Args:
        y1:      first NumPy array
        y2:      second NumPy array
        axis:    dimension(s) along which the correlation is computed. Any valid NumPy axis spec works here
        eps:     offset to the standard deviation to avoid exploding the correlation due to small division (default 1e-8)
        **kwargs: passed to final numpy.mean operation over standardized y1 * y2

    Returns: correlation array for the neurons
    """

    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=0) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=0) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)


def interpolate_signal_with_nans(time_input, signal_input, time_output, kind='linear'):
    """
    Fill NaNs in signal_input using interpolation, then interpolate onto time_output.

    Parameters:
    - time_input: 1D array of original time points
    - signal_input: 1D array of signal values at time_input (may contain NaNs)
    - time_output: 1D array of desired time points for interpolation
    - kind: interpolation kind ('linear', 'nearest', 'cubic', etc.)

    Returns:
    - signal_output: interpolated signal at time_output
    """
    # Ensure arrays are numpy arrays
    time_input = np.asarray(time_input)
    signal_input = np.asarray(signal_input)

    # Step 1: Interpolate over NaNs in signal_input
    valid = ~np.isnan(signal_input)
    if np.sum(valid) < 2:
        raise ValueError("Not enough valid points to interpolate.")

    fill_interp = interp1d(time_input[valid], signal_input[valid], kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    signal_filled = fill_interp(time_input)

    # Step 2: Interpolate to time_output
    output_interp = interp1d(time_input, signal_filled, kind=kind,
                             bounds_error=False, fill_value='extrapolate')
    signal_output = output_interp(time_output)

    return signal_output

def auprc_per_unit(
    y_true_np: np.ndarray,  # shape: (batch_size, num_units)
    y_pred_np: np.ndarray   # shape: (batch_size, num_units)
) -> np.ndarray:
    """
    Compute the Area Under the Precision-Recall Curve (AUPRC) for each unit (column) independently.

    Parameters:
    ----------
    y_true_np : np.ndarray
        A 2D NumPy array of binary ground truth labels (0 or 1),
        with shape (batch_size, number_of_units).

    y_pred_np : np.ndarray
        A 2D NumPy array of predicted probabilities (after sigmoid),
        with shape (batch_size, number_of_units).

    Returns:
    -------
    np.ndarray
        A 1D NumPy array of AUPRC scores for each unit.
        Units with no positive examples in `y_true_np` will have `np.nan`.
    """

    num_units = y_true_np.shape[1]
    auprc_scores = []

    for i in range(num_units):
        y_true_unit = y_true_np[:, i]
        y_pred_unit = y_pred_np[:, i]

        if np.sum(y_true_unit) == 0:
            # AUPRC is undefined when there are no positive samples
            auprc_scores.append(np.nan)
        else:
            score = average_precision_score(y_true_unit, y_pred_unit)
            auprc_scores.append(score)

    return np.array(auprc_scores)

def lowpass_filter(sig, cutoff=1.0, fs=100.0, order=4):
    """
    Apply a low-pass Butterworth filter to a 1D signal.

    Parameters
    ----------
    sig : np.ndarray
        Input 1D signal.
    cutoff : float
        Desired cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter (higher = sharper cutoff).

    Returns
    -------
    filtered : np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs                 # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, sig)
    return filtered


def generate_2d_rate_map(
    loc_xx: np.ndarray,  # 1D array of x-coordinates of the animal's location over time
    loc_yy: np.ndarray,  # 1D array of y-coordinates of the animal's location over time
    unit_spike_count: np.ndarray,  # 1D array of spike counts for one neuron at each time point
    loc_bin_width: float = 0.3,  # Spatial bin width for both x and y dimensions (in location units)
    dt: float = 0.1,  # Time interval between consecutive samples (in seconds)
):
    """
    Compute the 2D rate map of a neuron by binning the 2D location space and calculating 
    the firing rate (in spikes per second) as spike count divided by time spent in each spatial bin.

    Args:
        loc_xx (np.ndarray): 1D array of x-coordinates of the animal's position.
        loc_yy (np.ndarray): 1D array of y-coordinates of the animal's position.
        unit_spike_count (np.ndarray): 1D array of spike counts (e.g., 0, 1, 2) for a neuron at each time point.
        loc_bin_width (float): Width of spatial bins in both x and y directions.
        dt (float): Time interval (in seconds) between each recorded location sample.

    Returns:
        rate_map (np.ndarray): 2D array representing the firing rate (spikes per second) in each spatial bin.
        smoothed_rate_map (np.ndarray): Gaussian-smoothed version of the rate map.

    Example usage:
        from digitalbat.utility.utils import generate_2d_rate_map
        loc_xx = np.copy(all_behavior[0,:])
        loc_yy = np.copy(all_behavior[1,:])
        unit_spike_count = all_spike[249,:]
        loc_bin_width = 0.3 # the bin width for the location space
        dt = np.copy(bin_width)  # time interval for each data point (in seconds)

        rate_map, smoothed_rate_map = generate_2d_rate_map(
            loc_xx=loc_xx,
            loc_yy=loc_yy,
            unit_spike_count=unit_spike_count,
            loc_bin_width=loc_bin_width,
            dt=dt,
        )

        # Plot both unsmoothed and smoothed maps
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        im0 = axs[0].imshow(rate_map.T, origin='lower',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            aspect='auto', cmap='hot_r')
        axs[0].set_title("Unsmoothed Rate Map")
        axs[0].set_xlabel("X position")
        axs[0].set_ylabel("Y position")
        plt.colorbar(im0, ax=axs[0], label="spikes/s")
        im1 = axs[1].imshow(smoothed_rate_map.T, origin='lower',
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                            aspect='auto', cmap='hot_r')
        axs[1].set_title("Smoothed Rate Map (Gaussian)")
        axs[1].set_xlabel("X position")
        axs[1].set_ylabel("Y position")
        plt.colorbar(im1, ax=axs[1], label="spikes/s")
        plt.tight_layout()
    """

    # Define bin edges in x and y directions
    x_edges = np.arange(np.min(loc_xx), np.max(loc_xx) + loc_bin_width, loc_bin_width)
    y_edges = np.arange(np.min(loc_yy), np.max(loc_yy) + loc_bin_width, loc_bin_width)
    # Compute occupancy: number of time points spent in each spatial bin
    occupancy, _, _ = np.histogram2d(loc_xx, loc_yy, bins=[x_edges, y_edges])
    # Convert occupancy count to time (in seconds)
    occupancy_time = occupancy * dt
    # Compute total spike count in each bin (weighted histogram)
    spike_map, _, _ = np.histogram2d(loc_xx, loc_yy, bins=[x_edges, y_edges], weights=unit_spike_count)
    # Compute firing rate: spike count / occupancy time
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.divide(spike_map, occupancy_time)
        rate_map[np.isnan(rate_map)] = 0  # Replace NaNs (from divide-by-zero) with 0

    # Apply Gaussian smoothing to the rate map
    sigma = 1.0  # Standard deviation of Gaussian kernel (in units of spatial bins)
    smoothed_rate_map = gaussian_filter(rate_map, sigma=sigma)

    return x_edges, y_edges, rate_map, smoothed_rate_map


def poisson_fraction_deviance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the fraction deviance explained for a Poisson GLM.
    
    Parameters:
    - y_true: 1D array of observed counts (non-negative integers)
    - y_pred: 1D array of predicted rates (must be > 0)
    
    Returns:
    - Fraction of deviance explained (float between -inf and 1)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid log(0) and division by zero
    eps = np.finfo(float).eps
    y_pred = np.clip(y_pred, eps, None)
    # Null model prediction: mean of y_true
    y_mean = np.mean(y_true)
    y_mean = np.clip(y_mean, eps, None)

    def deviance(y, mu):
        term = np.zeros_like(y, dtype=float)
        nonzero = y > 0
        term[nonzero] = y[nonzero] * np.log(y[nonzero] / mu[nonzero])
        return 2 * np.sum(term - (y - mu))

    dev_model = deviance(y_true, y_pred)
    dev_null = deviance(y_true, np.full_like(y_true, y_mean))

    # Return fraction deviance explained
    return 1 - dev_model / (dev_null + 1e-6 )


# ---------------------------------------------------------------------
# Utility function: Arc length computation
# ---------------------------------------------------------------------
def arclength(x, y, z):
    """
    Compute arc length of a 3D curve using linear interpolation.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of the curve.

    Returns
    -------
    total_length : float
        Total length of the curve.
    seglen : np.ndarray
        Length of each segment (starting with 0 for the first point).
    """
    diffs = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    seglen = np.concatenate(([0], diffs))
    return np.sum(diffs), seglen


# ---------------------------------------------------------------------
# Utility function: Exact Fréchet distance between two curves
# ---------------------------------------------------------------------
def frechet_distance(P, Q):
    """
    Compute the discrete Fréchet distance between two curves P and Q.

    Parameters
    ----------
    P, Q : np.ndarray of shape (m, d) and (n, d)
        Curves represented as sequences of points in d-dimensional space.

    Returns
    -------
    dist : float
        The Fréchet distance.
    """
    # Cache for dynamic programming
    ca = np.full((len(P), len(Q)), -1.0)

    def _c(i, j):
        """Recursive helper function."""
        if ca[i, j] > -1:
            return ca[i, j]
        d = np.linalg.norm(P[i] - Q[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(i - 1, j),
                               _c(i - 1, j - 1),
                               _c(i, j - 1)), d)
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    return _c(len(P) - 1, len(Q) - 1)


# ---------------------------------------------------------------------
# Main class for clustering flights
# ---------------------------------------------------------------------
class FlightClusters:
    def __init__(self, r_vector, f_vector, Fs, 
                 ds_clus=7, pca_features=False, frechet_clus=False, k_means=False, 
                 alpha=1.1, reassign=True, N_min=5, ignore_z=False, calc_length=False):
        """
        Parameters
        ----------
        r_vector : array-like, shape (time, 3)
            3D positions over time.
        f_vector : array-like, shape (time,)
            Binary flying/not flying indicator (1 = flying, 0 = not flying).
        Fs : float
            Sampling frequency (Hz).
        ds_clus : int
            Number of points to downsample each trajectory for clustering.
        pca_features : bool
            Whether to apply PCA before clustering.
        frechet_clus : bool
            If True, use Fréchet distance for clustering.
        k_means : bool
            If True, use K-means clustering instead of hierarchical clustering.
        alpha : float
            Linkage distance threshold for hierarchical clustering.
        reassign : bool
            Placeholder for potential re-assignment logic.
        N_min : int
            Minimum number of flights per cluster (not enforced in current version).
        ignore_z : bool
            If True, ignore Z-coordinate in clustering.
        calc_length : bool
            If True, compute trajectory lengths and durations.
        """
        self.r_vector = np.asarray(r_vector)
        self.f_vector = np.asarray(f_vector).astype(int)
        self.Fs = Fs
        self.ds_clus = ds_clus
        self.pca_features = pca_features
        self.frechet_clus = frechet_clus
        self.k_means = k_means
        self.alpha = alpha
        self.reassign = reassign
        self.N_min = N_min
        self.ignore_z = ignore_z
        self.calc_length = calc_length

        self.results = {}

    def compute(self):
        """Main computation method: segment flights, downsample, cluster."""
        r, f, Fs = self.r_vector, self.f_vector, self.Fs

        # -----------------------------------------------------------------
        # Flight segmentation: find start and stop indices for each flight
        # -----------------------------------------------------------------
        num_flights = np.sum(np.diff(f) > 0)
        f_start = np.where(np.diff(f) > 0)[0] + 1
        f_stop = np.where(np.diff(f) < 0)[0]

        # Velocity vector and magnitude
        v = np.vstack(([0, 0, 0], np.diff(r, axis=0) * Fs))
        v_abs = np.linalg.norm(v, axis=1)

        # Allocate arrays for storing flight data
        max_len = np.max(f_stop - f_start) + 1
        all_flights = np.full((3, max_len, num_flights), np.nan)
        all_flights_vel = np.full((1, max_len, num_flights), np.nan)
        all_flights_ds = np.full((3, self.ds_clus, num_flights), np.nan)

        # -----------------------------------------------------------------
        # Extract and downsample each flight
        # -----------------------------------------------------------------
        for nf in range(num_flights):
            idx = slice(f_start[nf], f_stop[nf] + 1)
            traj = r[idx, :].T  # shape (3, n_points)
            vel = v_abs[idx]

            # Store full trajectories
            all_flights[:, :traj.shape[1], nf] = traj
            all_flights_vel[0, :len(vel), nf] = vel

            # Downsample to fixed number of points
            interp_idx = np.linspace(0, traj.shape[1] - 1, self.ds_clus)
            for dim in range(3):
                f_interp = interp1d(np.arange(traj.shape[1]), traj[dim, :], kind='linear')
                all_flights_ds[dim, :, nf] = f_interp(interp_idx)

        # -----------------------------------------------------------------
        # Prepare features for clustering
        # -----------------------------------------------------------------
        if not self.frechet_clus:
            # Flatten downsampled data: shape (num_flights, ds_clus*3)
            X = np.reshape(np.transpose(all_flights_ds, (2, 1, 0)), (num_flights, -1))
        else:
            # Keep the 3D structure for Fréchet distance
            X = np.transpose(all_flights_ds, (2, 1, 0))  # shape: (num_flights, ds_clus, 3)

        # Optionally remove Z-coordinate
        if self.ignore_z:
            if self.frechet_clus:
                X[:, :, 2] = 0
            else:
                X[:, 2::3] = 0

        # Optionally reduce dimensionality with PCA
        if self.pca_features and not self.frechet_clus:
            pca = PCA(n_components=4)
            X = pca.fit_transform(X)

        # -----------------------------------------------------------------
        # Clustering
        # -----------------------------------------------------------------
        if self.k_means and not self.frechet_clus:
            idx = KMeans(n_clusters=15).fit_predict(X)
        else:
            if not self.frechet_clus:
                # Standard distance: standardized Euclidean
                weights = np.tile(np.ones(self.ds_clus), 3)
                Y = pdist(X, metric='seuclidean', V=weights)
                Y /= self.ds_clus
            else:
                # Compute full pairwise Fréchet distance matrix
                n = X.shape[0]
                dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = frechet_distance(X[i], X[j])
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
                Y = squareform(dist_matrix)

            # Hierarchical clustering
            Z = linkage(Y, method='single')
            idx = fcluster(Z, t=self.alpha, criterion='distance')

        # -----------------------------------------------------------------
        # Build results dictionary
        # -----------------------------------------------------------------
        self.results = {
            'id': idx,
            'start_frame': f_start,
            'stop_frame': f_stop,
            'pos': all_flights,
            'vel': all_flights_vel,
            'Fs': Fs,
            'N': num_flights,
            'ds_pos': all_flights_ds
        }

        # Optionally compute flight lengths and durations
        if self.calc_length:
            lengths = np.zeros(num_flights)
            durs = np.zeros(num_flights)
            for i in range(num_flights):
                mask = ~np.isnan(all_flights[0, :, i])
                traj = all_flights[:, mask, i].T
                length, seglen = arclength(traj[:, 0], traj[:, 1], traj[:, 2])
                lengths[i] = length
                durs[i] = (f_stop[i] - f_start[i]) / Fs
            self.results['length'] = lengths
            self.results['dur'] = durs

        return self.results

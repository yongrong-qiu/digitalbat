import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score
from scipy.ndimage import gaussian_filter
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

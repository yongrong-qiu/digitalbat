import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
import torch
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from numpy.typing import ArrayLike
from typing import Tuple, Union, Callable


def bat_palette(palette='tab10'):
    """Return a `{bat_id: rgb}` dict in canonical project order.

    Bats are extracted from `DEFAULT_SESSION_NAMES` (third underscore-split
    token).  Pass into seaborn via `palette=bat_palette()` with `hue='bat'`
    to keep colors consistent across notebooks/figures.
    """
    import seaborn as sns
    from ..datasets.defaults import DEFAULT_SESSION_NAMES
    seen = []
    for s in DEFAULT_SESSION_NAMES:
        b = s.split('_')[2]
        if b not in seen:
            seen.append(b)
    colors = sns.color_palette(palette, n_colors=max(len(seen), 3))
    return {b: c for b, c in zip(seen, colors)}


def parse_bat(session_name):
    """`Dataset_3_14543_240419` -> `'14543'`."""
    return session_name.split('_')[2]


def wilcoxon_safe(diff):
    """Two-sided Wilcoxon signed-rank on `diff`; returns (statistic, pvalue).

    Handles empty / all-zero / all-NaN inputs by returning (nan, nan) so
    callers can branch on `np.isfinite(p)` without try/except.
    """
    from scipy.stats import wilcoxon
    diff = np.asarray(diff)
    diff = diff[np.isfinite(diff)]
    if diff.size < 1 or np.all(diff == 0):
        return np.nan, np.nan
    res = wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
    return float(res.statistic), float(res.pvalue)


def stars_for_p(p):
    """Three-tier stars: '*' p<.05, '**' p<.01, '***' p<.001.  '' for n.s. or nan."""
    if p is None or not np.isfinite(p):
        return ''
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    return ''


def add_significance_bracket(ax, x1, x2, p, *, y=None, height_frac=0.03,
                              color='k', linewidth=1.0, fontsize=12):
    """Draw a horizontal bracket from `x1` to `x2` with stars above; no-op when p >= .05.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x1, x2 : float
        Bracket endpoints in data coordinates (e.g. categorical positions 0, 1).
    p : float
        p-value; if >= .05 or non-finite, this function does nothing.
    y : float, optional
        Bracket baseline (data coords).  If None, placed just above the data
        in the bracket span via `ax.dataLim`.
    height_frac : float
        Stem height as a fraction of the y-range.
    """
    stars = stars_for_p(p)
    if not stars:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    if y is None:
        y = ymax - 0.05 * span
    h = height_frac * span
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=linewidth, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, stars, ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold', color=color, clip_on=False)
    # Make sure stars are visible.
    ax.set_ylim(top=max(ymax, y + h + 0.08 * span))


def annotate_pvalue_corner(ax, p, n=None, *, loc='upper right',
                            fontsize=10, color='k'):
    """Annotate a corner of `ax` with stars (+ optional p, n) when significant.

    No-op if p >= .05 or non-finite, per the project convention to hide n.s.
    """
    stars = stars_for_p(p)
    if not stars:
        return
    bits = [stars]
    if np.isfinite(p):
        bits.append(f'p={p:.1e}')
    if n is not None:
        bits.append(f'n={n}')
    text = '\n'.join(bits)
    xy_map = {
        'upper right': (0.97, 0.97, 'right', 'top'),
        'upper left':  (0.03, 0.97, 'left',  'top'),
        'lower right': (0.97, 0.03, 'right', 'bottom'),
        'lower left':  (0.03, 0.03, 'left',  'bottom'),
    }
    x, y, ha, va = xy_map[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.85))


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
    loc_xx_lim: np.ndarray = None, # boudary of x dimensions, 2-element array 
    loc_yy_lim: np.ndarray = None, # boudary of x dimensions, 2-element array 
    loc_bin_width: float = 0.1,  # Spatial bin width for both x and y dimensions (in location units)
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
    if loc_xx_lim is not None and loc_yy_lim is not None:
        x_edges = np.arange(loc_xx_lim[0], loc_xx_lim[1] + loc_bin_width, loc_bin_width)
        y_edges = np.arange(loc_yy_lim[0], loc_yy_lim[1] + loc_bin_width, loc_bin_width)
    else:
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

def plot_2d_rate_map(
    unit: int, # neuron index
    target_type: int, # flight cluster type index
    loc_xx: np.ndarray,  # 1D array of x-coordinates of the animal's location over time
    loc_yy: np.ndarray,  # 1D array of y-coordinates of the animal's location over time
    spike_count_val: np.ndarray,  # 1D array of spike counts for one neuron at each time point
    spike_count_pred: np.ndarray,  # 1D array of spike counts for one neuron at each time point
    loc_xx_lim: np.ndarray = None, # boudary of x dimensions, 2-element array 
    loc_yy_lim: np.ndarray = None, # boudary of x dimensions, 2-element array 
    loc_bin_width: float = 0.1,  # Spatial bin width for both x and y dimensions (in location units)
    dt: float = 0.1,  # Time interval between consecutive samples (in seconds)
    colorbar_label: str = "Spikes/s", # label for the colorbar
):
    """
    For a specific "unit" and a specific "target_type", plot the 2d rate map for the recorded and predicted data
    """
    x_edges, y_edges, rate_map, smoothed_rate_map = generate_2d_rate_map(
        loc_xx=loc_xx,
        loc_yy=loc_yy,
        unit_spike_count=spike_count_val,
        loc_xx_lim=loc_xx_lim,
        loc_yy_lim=loc_yy_lim,
        loc_bin_width=loc_bin_width,
        dt=dt,
    )
    #    
    _, _, _, smoothed_rate_map_pred = generate_2d_rate_map(
        loc_xx=loc_xx,
        loc_yy=loc_yy,
        unit_spike_count=spike_count_pred,
        loc_xx_lim=loc_xx_lim,
        loc_yy_lim=loc_yy_lim,
        loc_bin_width=loc_bin_width,
        dt=dt,
    )
    # Plot smoothed maps
    vmin = np.min(np.concatenate((smoothed_rate_map, smoothed_rate_map_pred), ))
    vmax = np.max(np.concatenate((smoothed_rate_map, smoothed_rate_map_pred), ))
    fig, axs = plt.subplots(1, 2, figsize=(5, 2))
    im0 = axs[0].imshow(smoothed_rate_map.T, origin='lower', vmin=vmin, vmax=vmax,
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        aspect='auto', cmap='hot_r')
    axs[0].set_title(f"Unit: {unit}; type: {target_type}; recorded")
    axs[0].set_xlabel("X position")
    axs[0].set_ylabel("Y position")
    plt.colorbar(im0, ax=axs[0], label=colorbar_label)
    im1 = axs[1].imshow(smoothed_rate_map_pred.T, origin='lower', vmin=vmin, vmax=vmax,
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        aspect='auto', cmap='hot_r')
    axs[1].set_title("Predicted")
    axs[1].set_xlabel("X position")
    # axs[1].set_ylabel("Y position")
    plt.colorbar(im1, ax=axs[1], label=colorbar_label)
    plt.tight_layout()

def plot_2d_trajectory_response(
    unit: int, # neuron index
    target_type: int, # flight cluster type index
    loc_xx: np.ndarray,  # 1D array of x-coordinates of the animal's location over time
    loc_yy: np.ndarray,  # 1D array of y-coordinates of the animal's location over time
    spike_count_val: np.ndarray,  # 1D array of spike counts for one neuron at each time point
    spike_count_pred: np.ndarray,  # 1D array of spike counts for one neuron at each time point
    loc_xx_lim: np.ndarray = None, # boudary of x dimensions, 2-element array 
    loc_yy_lim: np.ndarray = None, # boudary of x dimensions, 2-element array 
    colorbar_label: str = "Spike count", # label for the colorbar
):
    """
    2D trajectory with colors indicating neural responses along the path
    For a specific "unit" and a specific "target_type", plot the 2d trajectory for the recorded and predicted data
    """
    # Make line segments between successive points
    points = np.array([loc_xx, loc_yy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Normalize spike counts for colormap
    norm = plt.Normalize(np.stack([spike_count_val, spike_count_pred]).min(), np.stack([spike_count_val, spike_count_pred]).max())
    # Create a line collection with color mapping
    lc_val = LineCollection(segments, cmap='hot', norm=norm)
    lc_val.set_array(spike_count_val)
    lc_val.set_linewidth(1)
    lc_pred = LineCollection(segments, cmap='hot', norm=norm)
    lc_pred.set_array(spike_count_pred)
    lc_pred.set_linewidth(1)
    # plot
    fig, axs = plt.subplots(1, 2, figsize=(5, 2))
    line = axs[0].add_collection(lc_val)
    axs[0].set_title(f"Unit: {unit}; type: {target_type}; recorded")
    axs[0].set_xlim(loc_xx_lim)
    axs[0].set_ylim(loc_yy_lim)
    axs[0].set_xlabel("X position")
    axs[0].set_ylabel("Y position")
    plt.colorbar(line, ax=axs[0], label=colorbar_label)
    line = axs[1].add_collection(lc_pred)
    axs[1].set_title("Predicted")
    axs[1].set_xlabel("X position")
    axs[1].set_xlim(loc_xx_lim)
    axs[1].set_ylim(loc_yy_lim)
    # axs[1].set_ylabel("Y position")
    plt.colorbar(line, ax=axs[1], label=colorbar_label)
    plt.tight_layout()
    return None


def plot_2d_trajectory_response_multiple(
    session_target, # a string specifying the session name
    units: np.ndarray, # 1D array of neuronal indices
    X_val_np: np.ndarray, # shape: (batch_size, time_point, 3)
    y_val_np: np.ndarray, # shape: (batch_size, time_point, neurons)
    y_pred_np: np.ndarray, # shape: (batch_size, time_point, neurons)
    mask_val: np.ndarray, # shape: (batch_size, time_point)
    flight_type_index: np.ndarray, # shape: (batch_size, ), flight cluster type for each trajectory
    meta_info_dict, # dictionary with keys 'behavior_std', 'behavior_mean', 'r_lim' for experimental sessions
    plot_flight_type_num = None, # the number of flight cluster types to be visualized
):
    """
    Similar to plot_2d_trajectory_response() but for multiple units and flight cluster types in a session 'session_target'
    2D trajectory with colors indicating neural responses along the path, for the recorded and predicted data

    X_val_np.shape: (48, 70, 3), y_val_np.shape: (48, 70, 95), y_pred_np.shape: (48, 70, 95), mask_val.shape: (48, 70), flight_type_index.shape: (48,)
    """
    flight_type_index = flight_type_index.astype(int)
    target_types_indomain = np.array(meta_info_dict['flight_type_test_indomain'][session_target], dtype=int)
    target_types_ood = np.array(meta_info_dict['flight_type_test_ood'][session_target], dtype=int)
    if plot_flight_type_num:
        nrows, ncols = plot_flight_type_num, len(units)*2
    else:
        nrows, ncols = len(np.unique(flight_type_index)), len(units)*2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*1.5))
    for ii,target_type in enumerate(np.concatenate((target_types_indomain, target_types_ood))):
        if plot_flight_type_num:
            if ii >= plot_flight_type_num:
                break
        target_ind = np.where(flight_type_index == target_type)[0]
        #
        target_X_val = X_val_np[target_ind, :, :]
        target_y_val = y_val_np[target_ind, :, :]
        target_y_pred = y_pred_np[target_ind, :, :]
        target_mask_val = mask_val[target_ind, ]
        #
        target_X_val = target_X_val[target_mask_val == 1.0]
        target_y_val = target_y_val[target_mask_val == 1.0]
        target_y_pred = target_y_pred[target_mask_val == 1.0]
        # print(f'target_X_val.shape: {target_X_val.shape}, target_y_val.shape: {target_y_val.shape}, target_y_pred.shape: {target_y_pred.shape}')
        # target_X_val.shape: (1512, 3), target_y_val.shape: (1512, 168), target_y_pred.shape: (1512, 168) # 168 neurons
        loc_xx = target_X_val[:, 0] * meta_info_dict['behavior_std'][session_target][0, 0, 0] + meta_info_dict['behavior_mean'][session_target][0, 0, 0]
        loc_yy = target_X_val[:, 1] * meta_info_dict['behavior_std'][session_target][0, 0, 1] + meta_info_dict['behavior_mean'][session_target][0, 0, 1]
        loc_xx_lim = meta_info_dict['r_lim'][session_target][0, :]
        loc_yy_lim = meta_info_dict['r_lim'][session_target][1, :]
        for jj, unit in enumerate(units):
            spike_count_val = target_y_val[:, unit]
            spike_count_pred = target_y_pred[:, unit]
            # Make line segments between successive points
            points = np.array([loc_xx, loc_yy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Normalize spike counts for colormap
            norm = plt.Normalize(np.stack([spike_count_val, spike_count_pred]).min(), np.stack([spike_count_val, spike_count_pred]).max())
            # Create a line collection with color mapping
            lc_val = LineCollection(segments, cmap='hot', norm=norm)
            lc_val.set_array(spike_count_val)
            lc_val.set_linewidth(1)
            lc_pred = LineCollection(segments, cmap='hot', norm=norm)
            lc_pred.set_array(spike_count_pred)
            lc_pred.set_linewidth(1)
            # plot
            line = axs[ii,jj*2].add_collection(lc_val)
            if ii == 0:
                axs[ii,jj*2].set_title(f"Unit: {unit}; recorded")
            axs[ii,jj*2].set_xlim(loc_xx_lim)
            axs[ii,jj*2].set_ylim(loc_yy_lim)
            if ii == nrows-1:
                axs[ii,jj*2].set_xlabel("X")
            if jj == 0:
                label = f"Y; type: {target_type}" if target_type in target_types_indomain else f"Y; type: {target_type}; OOD"
                axs[ii,jj*2].set_ylabel(label)
            plt.colorbar(line, ax=axs[ii,jj*2])
            line = axs[ii,jj*2+1].add_collection(lc_pred)
            if ii == 0:
                axs[ii,jj*2+1].set_title("Predicted")
            # axs[ii,jj*2+1].set_xlabel("X position")
            axs[ii,jj*2+1].set_xlim(loc_xx_lim)
            axs[ii,jj*2+1].set_ylim(loc_yy_lim)
            # axs[1].set_ylabel("Y position")
            if jj == len(units):
                plt.colorbar(line, ax=axs[ii,jj*2+1], label="Spike count")
            else:
                plt.colorbar(line, ax=axs[ii,jj*2+1])
            plt.tight_layout()
    return None


def plot_3d_trajectory(
    trajectory_3d, # numpy array, shape (T, 3)
    loc_lim, # numpy array, shape (3, 2)
    flight_cluster_type: int = 1, # the cluster type of the flight path, such as: 1, 2, 3, etc.
    time_bin_width = 0.1, # bin_width for each time point in trajectory_3d
    visualize_array = None, # None or a numpy array with a shape (T,)
    cmap = 'viridis', # colormap
    set_zero_as_medium = False, # When using the colormap 'bwr', if we want to set 'zero' to the white color, we need to set 'zero' as medium value
    cax_ylabel = 'Time [s]', # colorbar ylabel
):
    """
    For a flight path, plot a 3D trajectory as a continuous line, with color indicating time
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    T = trajectory_3d.shape[0]
    points = np.copy(trajectory_3d)
    # Create line segments between consecutive points
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
    
    if visualize_array is None:
        # Time array for coloring
        visualize_array = np.linspace(0, T*time_bin_width, T - 1)
    else:
        visualize_array = visualize_array[:T-1]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colored 3D line
    if set_zero_as_medium:
        vmax = np.max(np.abs(visualize_array))
        vmin = -1.0 * vmax
    else:
        vmax, vmin = np.max(visualize_array), np.min(visualize_array)
    lc = Line3DCollection(segments, array=visualize_array, cmap=cmap, linewidth=2)
    lc.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(lc)

    ax.scatter(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], c='k', s=3)
    # start of a flight path
    ax.scatter(*trajectory_3d[0,:3], marker='*', c='red')
    # Set limits
    ax.set_xlim(loc_lim[0])
    ax.set_ylim(loc_lim[1])

    # TODO: For some sessions, such as Dataset_1_32622_231009, 1st flight path in test data
    # the z limits offered by 'loc_lim' is zero, but 'trajectory_3d' has negative values in z-axis
    if trajectory_3d[:,2].min() > loc_lim[2,0]:
        ax.set_zlim(loc_lim[2])
    else:
        ax.set_zlim([-loc_lim[2,1], loc_lim[2,1]])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D trajectory for Type {flight_cluster_type}")
    # Colorbar
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.9, 0.2, 0.02, 0.3])  # Adjust to fit within bounds
    fig.colorbar(lc, cax=cax)
    cax.set_ylabel(cax_ylabel)
    # cbar = fig.colorbar(lc, ax=ax, shrink=0.5)
    # cbar.set_label("Time [s]")
    return None


def plot_trajectory_saliency_ax(ax, trajectory, saliency_matrix, t_resp_idx, loc_xyz_lim):
    """
    trajectory: (90, 3) array of (x, y, z)
    saliency_matrix: (90, 90) array (Response Time x Location Time)
    t_resp_idx: The specific time point of the neural response to analyze
    """
    
    # 1. Extract the saliency vector for the chosen response time
    # This tells us how much each point in the trajectory (t') 
    # contributed to the response at t_resp_idx.
    weights = saliency_matrix[t_resp_idx, :] 
    # # 2. Setup the 3D plot
    # fig = plt.figure(figsize=(3, 2))
    # ax = fig.add_subplot(111, projection='3d')
    # 3. Create segments for the 3D LineCollection
    # Each segment is a pair of points: [(x1,y1,z1), (x2,y2,z2)]
    points = trajectory.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # 4. Create the 3D LineCollection with a colormap
    norm = plt.Normalize(weights.min(), weights.max())
    lc = Line3DCollection(segments, cmap='hot', norm=norm)
    # Set the values used for colormapping to our saliency weights
    lc.set_array(weights)
    lc.set_linewidth(2)
    # 5. Add to plot
    line = ax.add_collection3d(lc)
    # 6. Highlight the bat's actual position at t_resp_idx
    curr_pos = trajectory[t_resp_idx]
    ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], 
               color='cyan', s=50, label='Current Pos.', edgecolors='k')
    # Plotting adjustments
    if loc_xyz_lim is not None:
        ax.set_xlim(loc_xyz_lim[0,:])
        ax.set_ylim(loc_xyz_lim[1,:])
        ax.set_zlim(loc_xyz_lim[2,:])
        # Calculate the range for each axis
        # Set aspect ratio proportional to the data ranges
        ax.set_box_aspect((loc_xyz_lim[0,1] - loc_xyz_lim[0,0], loc_xyz_lim[1,1] - loc_xyz_lim[1,0], loc_xyz_lim[2,1] - loc_xyz_lim[2,0]))

        # wireframe box
        x_lim = loc_xyz_lim[0,:]
        y_lim = loc_xyz_lim[1,:]
        z_lim = loc_xyz_lim[2,:]
        # Bottom square
        ax.plot([x_lim[0], x_lim[1], x_lim[1], x_lim[0], x_lim[0]], 
                [y_lim[0], y_lim[0], y_lim[1], y_lim[1], y_lim[0]], 
                z_lim[0], color='k', lw=0.5)
        # Top square
        ax.plot([x_lim[0], x_lim[1], x_lim[1], x_lim[0], x_lim[0]], 
                [y_lim[0], y_lim[0], y_lim[1], y_lim[1], y_lim[0]], 
                z_lim[1], color='k', lw=0.5)
        # Vertical connectors
        for x in x_lim:
            for y in y_lim:
                ax.plot([x, x], [y, y], [z_lim[0], z_lim[1]], color='k', lw=0.5)
                
    ax.set_title(f'Saliency at t={t_resp_idx}')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Hide the three axis spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    # ax.set_title(f'Saliency Map for Response at t={t_resp_idx}')
    # fig.colorbar(line, ax=ax, label='Saliency')
    # plt.legend()
    # plt.show()

    return line # Return for colorbar if needed

def plot_trajectory_saliency(trajectory, saliency_matrix, t_resp_idx, loc_xyz_lim):
    """
    For a given trajectory and a neuron, plot the saliency for the response at a given time point w.r.t the trajectory
    trajectory: (90, 3) array of (x, y, z)
    saliency_matrix: (90, 90) array (Response Time x Location Time)
    t_resp_idx: a list of time points
    """
    # Configuration
    # t_points = [10, 40, 80] # The indices you want to show in a row
    n_plots = len(t_resp_idx)
    
    # Create the figure with multiple 3D subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(2.5 * n_plots, 2), subplot_kw={'projection': '3d'})
    # Loop through and plot
    for i, t_idx in enumerate(t_resp_idx):
        last_line = plot_trajectory_saliency_ax(axes[i], trajectory, saliency_matrix, t_idx, loc_xyz_lim)
    
    # Optional: Single colorbar for the whole row
    # fig.colorbar(last_line, ax=axes.ravel().tolist(), shrink=0.5, label='Saliency')
    # plt.show()
    return None


def poisson_fraction_deviance_explained(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the fraction deviance explained for a Poisson GLM.

    Parameters:
    - y_true: 1D array of observed counts (non-negative integers)
    - y_pred: 1D array of predicted rates (must be > 0)

    Returns:
    - Fraction of deviance explained (float between -inf and 1)

    Note: 
    We updated a bit to match the numbers exactly with https://github.com/sytseng/GLM_Tensorflow_2/blob/main/code/glm_class.py
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid log(0) and division by zero
    eps = 1e-33 # 1e-33 # np.finfo(float).eps
    y_pred = np.clip(y_pred, eps, None)
    # Null model prediction: mean of y_true
    y_true = np.clip(y_true, eps, None)
    y_mean = np.mean(y_true)
    y_mean = np.clip(y_mean, eps, None)

    def deviance(y, mu):
        # term = np.zeros_like(y, dtype=float)
        # nonzero = y > 0
        # term[nonzero] = y[nonzero] * np.log(y[nonzero] / mu[nonzero])
        return 2 * np.sum(y * (np.log(y) - np.log(mu)) - (y - mu))

    dev_model = deviance(y_true, y_pred)
    dev_null = deviance(y_true, np.full_like(y_true, y_mean))

    # Return fraction deviance explained
    return 1 - dev_model / (dev_null + eps) # 1e-6

def poisson_fde_population(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Use poisson_fraction_deviance_explained() to compute FDE for a population of neurons.

    Args:
    - y_true: recorded responses, 2D array with a shape of (TT, neuron_number)
    - y_pred: predicted responses, 2D array with a shape of (TT, neuron_number)

    Return:
    - 1D array, FDE for each neuron
    """

    fdes = np.empty((y_true.shape[1]))
    for ii in range(len(fdes)):
        fdes[ii] = poisson_fraction_deviance_explained(y_true[:,ii], y_pred[:,ii]) 
    
    return fdes


# Valid function: FDE only on well-fit neurons (FDE > 0)
def poisson_fde_population_filtered(y_true, y_pred):
    """
    Compute FDE for a population of neurons, only on well-fit neurons (FDE > 0)
    """
    fdes = poisson_fde_population(y_true, y_pred)
    return fdes[fdes > 0]  # Only neurons with FDE > 0

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


def compute_fractional_abscissa(path, eps=1e-12):
    """
    Compute the normalized arc-length parameter (fractional abscissa)
    for a 2D or 3D trajectory.

    Parameters
    ----------
    path : array of shape (T, D)
        The flight path, where T is the number of time points and
        D = 2 or 3 is the spatial dimension (x,y) or (x,y,z).

    Returns
    -------
    frac : array of shape (T,)
        The normalized arc-length values in the range [0, 1],
        where frac[0] = 0 at the starting point and frac[-1] = 1 at the end.

    Notes
    -----
    This function re-parameterizes the trajectory based on physical
    distance traveled rather than the original sampling times.
    """

    # 1. Compute differences between successive points along the path: Δx, Δy, (Δz)
    diffs = np.diff(path, axis=0)  # shape (T-1, D)

    # 2. Compute the Euclidean distance between each pair of successive points
    #    This gives segment lengths along the trajectory.
    lengths = np.linalg.norm(diffs, axis=1)  # shape (T-1,)

    # 3. Compute cumulative arc length
    #    cumlen[t] = total distance traveled from start to time t
    cumlen = np.concatenate([[0.0], np.cumsum(lengths)])  # shape (T,)

    total = cumlen[-1]
    if total < eps:
        # trajectory has no length at all
        return np.linspace(0, 1, len(cumlen))

    # Normalize by total path length so that final value = 1
    frac = cumlen / total

    # fix duplicates: enforce strict monotonicity
    # if frac[t] == frac[t-1], add tiny offsets
    for i in range(1, len(frac)):
        if frac[i] <= frac[i - 1]:
            frac[i] = frac[i - 1] + eps

    return frac


def resample_response(path, response, n=100):
    """
    Resample a neural response matrix along a normalized arc-length
    representation of a flight path.

    Parameters
    ----------
    path : array of shape (T, D)
        Original trajectory in 2D or 3D.
    response : array of shape (T, N)
        Neural responses for N neurons at T time points.
    n : int
        Number of resampled positions (uniform in arc length).

    Returns
    -------
    response_resampled : array of shape (n, N)
        Responses resampled on a uniform arc-length grid.

    Notes
    -----
    - Each neuron's response is treated as a function of arc length.
    - The interpolation is applied *independently* to each neuron.
    """

    # Number of time points (T) and neurons (N)
    T, N = response.shape

    # 1. Compute fractional arc-length coordinates for the original path
    frac = compute_fractional_abscissa(path)   # shape (T,)

    # 2. Create the target arc-length grid (uniform from 0 to 1)
    new_frac = np.linspace(0, 1, n)            # shape (n,)

    # 3. Create interpolator for all neurons at once
    #    axis=0 tells interp1d that interpolation happens along the time dimension
    #    Input shape: (T, N)
    interpolator = interp1d(frac, response, kind='linear', axis=0)

    # 4. Resample: output shape will be (n, N)
    response_resampled = interpolator(new_frac)

    return response_resampled


def nearest_neighbor_score(
    dataloaders,
    meta_info_dict,
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    """
    Nearest neighbor analysis.
    For each flight path in test data, we find the paths in the training/validation data with the same cluster type, we call them 'nearest-neighbor' paths.
    For each pair of test and nearest-neighbor paths, we resample each neuron's responses along arc-length representation of the paths. 
    We additionally average resampled responses across paths for nearest-neighbor flights.
    For each neuron, we concatenate the resampled responses across flight paths and compute the correlation between the test data and nearest-neighbor data. 

    valid_fn = lambda y_true, y_pred: corr(y_true, y_pred, axis=0)
    or:
    valid_fn = lambda y_true, y_pred: poisson_fde_population(y_true, y_pred)
    """
    
    nn_arrays_dict = {} # a dictionary of score arrays (for each unit) for different sessions
    session_targets = list(dataloaders['test'].keys()) # ['Dataset_1_32622_231010'] # list(dataloaders['test'].keys())
    for session_target in session_targets:
        print(f'session_target: {session_target}')
        for (session_name, session_data),(_, session_data2) in zip(dataloaders['train'].items(), dataloaders['valid'].items()):
            # print(session_name, session_data)
            if session_name == session_target:
                target_types = np.array(meta_info_dict['flight_type_test_indomain'][session_name], dtype=int) # data type: int
                nn_response = {}
                for batch_id, batch in enumerate(session_data): # for flights in training data
                    X_val = batch[0] # (torch.Size([15, 70, 3]),
                    y_val = batch[1] # torch.Size([15, 70, 54])
                    mask_val = batch[2] # torch.Size([15, 70]))
                    flight_type_index = batch.flight_type_index.detach().cpu().numpy().astype(int) # data type: int
                    # Find indices where flight_type_index ∈ target_inds, here we loop through each target_type
                    for ii,target_type in enumerate(target_types):
                        target_indices = np.where(flight_type_index == target_types[ii])[0]
                        # Each path has slightly different time points, we just use the minimum one
                        temp_path_time = int(np.min(np.sum(mask_val.detach().cpu().numpy()[target_indices], axis=1)))
                        resample_resps = [] # a list of resampled responses for each flight path
                        for jj in range(len(target_indices)): # loop through each flight path for the particular target_type
                            mask = mask_val.detach().cpu().numpy()[target_indices[jj]]   
                            path = X_val.detach().cpu().numpy()[target_indices[jj]][mask == 1.0]
                            resp = y_val.detach().cpu().numpy()[target_indices[jj]][mask == 1.0]
                            resample_resps.append(resample_response(path, resp, n=temp_path_time))
                            # if target_type == 10:
                                # print(f'target_type: {target_type}, temp_path_time: {temp_path_time}')
                        nn_response[str(target_type)] = resample_resps
                        
                for batch_id, batch in enumerate(session_data2): # for flights in validation data
                    X_val = batch[0] # (torch.Size([15, 70, 3]),
                    y_val = batch[1] # torch.Size([15, 70, 54])
                    mask_val = batch[2] # torch.Size([15, 70]))
                    flight_type_index = batch.flight_type_index.detach().cpu().numpy().astype(int) # data type: int
                    # Find indices where flight_type_index ∈ target_inds, here we loop through each target_type
                    for ii,target_type in enumerate(target_types):
                        target_indices = np.where(flight_type_index == target_types[ii])[0]
                        # resample_resps = [] # a list of resampled responses for each flight path
                        for jj in range(len(target_indices)): # loop through each flight path for the particular target_type
                            mask = mask_val.detach().cpu().numpy()[target_indices[jj]]   
                            path = X_val.detach().cpu().numpy()[target_indices[jj]][mask == 1.0]
                            resp = y_val.detach().cpu().numpy()[target_indices[jj]][mask == 1.0]
                            # resample_resps.append(resample_response(path, resp, n=temp_path_time))
                            nn_response[str(target_type)].append(resample_response(path, resp, n=nn_response[str(target_type)][0].shape[0]))
                            # if target_type == 10:
                                # print(f'target_type: {target_type}, temp_path_time: {nn_response[str(target_type)][0].shape[0]}')
                        nn_response[str(target_type)] = np.array(nn_response[str(target_type)])
                        # print(np.array(nn_response[str(target_type)]).shape) # (5, 31, 95): 5 repeats 31 time points 95 neurons; (1, 36, 95); ...

        # get the flight path and resample responses in test data
        for session_name, session_data in dataloaders['test'].items():
            # print(session_name, session_data)
            if session_name == session_target:
                test_response = {}
                for batch_id, batch in enumerate(session_data): # only for 1 batch 
                    X_val = batch[0] # X_val.shape: torch.Size([48, 70, 3])
                    y_val = batch[1] # y_val.shape: torch.Size([48, 70, 95])
                    mask_val = batch[2] # mask_val.shape: torch.Size([48, 70]) (batch_size, time_frames)
                    flight_type_index = batch.flight_type_index.detach().cpu().numpy().astype(int) # data type: int
                    # Find indices where flight_type_index ∈ target_inds, here we loop through each target_type
                    for ii,target_type in enumerate(target_types):
                        target_indices = np.where(flight_type_index == target_types[ii])[0]
                        resample_resps = [] # a list of resampled responses for each flight path
                        for jj in range(len(target_indices)): # loop through each flight path for the particular target_type
                            mask = mask_val.detach().cpu().numpy()[target_indices[jj]]   
                            path = X_val.detach().cpu().numpy()[target_indices[jj]][mask == 1.0]
                            resp = y_val.detach().cpu().numpy()[target_indices[jj]][mask == 1.0]
                            resample_resps.append(resample_response(path, resp, n=nn_response[str(target_type)][0].shape[0]))
                            # if target_type == 10:
                                # print(f'target_type: {target_type}, temp_path_time: {temp_path_time}')
                        test_response[str(target_type)] = np.array(resample_resps)
                        # print(np.array(resample_resps).shape) # (5, 31, 95): 5 repeats 31 time points 95 neurons; (1, 36, 95); ...

        # average responses across flight paths belong to the same cluster
        for key,value in nn_response.items():
            nn_response[key] = np.mean(value, axis=0)
        # expand arrays in nn_response have the same shape with test_response
        for key,value in nn_response.items():
            nn_response[key] = np.repeat(value[None, :, :], test_response[key].shape[0], axis=0)
            # print(nn_response[key].shape)
        # concatenate responses across cluster types
        nn_response = np.vstack([np.reshape(resp, (-1, resp.shape[2])) for resp in list(nn_response.values())])
        test_response = np.vstack([np.reshape(resp, (-1, resp.shape[2])) for resp in list(test_response.values())])
        print(f'nn_response.shape: {nn_response.shape}, test_response.shape: {test_response.shape}') # nn_response.shape: (1619, 95), test_response.shape: (1619, 95)
        nn_scores = valid_fn(test_response, nn_response)
        nn_arrays_dict[session_target] = nn_scores

    return nn_arrays_dict, nn_response, test_response


def set_random_state(seed):
    """
    Set random seed for pytorch
    """
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)    # reproducible
    return None


def compute_shift_attention(attn):
    """
    For each time point, we reindex the attention matrix by a specific temporal offset relative to the query position, 
    producing a time-lag importance profile that quantifies how strongly each position attends to the past, present, and future. 
    This transformation enables direct comparison of temporal dependencies across all positions in the sequence.

    attn: attention weight matrix, numpy array of shape (T, T)
    returns shift_scores: (T, 2T-1), where each column corresponds to a time-lag Δ
    """
    T = attn.shape[0]
    max_shift = T - 1
    num_shifts = 2*T - 1

    # Prepare output
    shift_scores = np.zeros((T, num_shifts), dtype=attn.dtype)

    # Compute all pairwise Δ = j - i
    i = np.arange(T)[:, None]   # shape (T,1)
    j = np.arange(T)[None, :]   # shape (1,T)
    diff = j - i                # (T,T), values from -(T-1) … (T-1)

    # Shift diff to valid array indices
    col_idx = diff + max_shift  # now in 0 … 2T-2

    # "Scatter add" into shift_scores
    for r in range(T):
        shift_scores[r].put(col_idx[r], attn[r])

    return shift_scores


class OrthogonalityPermutationTest:
    """
    Orthogonality permutation test for neurons within one experimental session.

    Example usage:
        tester = OrthogonalityPermutationTest(
            X=readout_weight_sessions,
            df=inds_sessions_df,
            session_name=list(dataloaders['test'].keys())[4],
            n_perm=1000,
            random_state=0,
        )
        sig_df = tester.significant_pairs_with_session_indices(alpha=0.05)
        row = sig_df.iloc[20]
        tester.plot_null_vs_observed(
            session_neuron_idx_1=row["session_neuron_idx_1"],
            session_neuron_idx_2=row["session_neuron_idx_2"],
        )
    """
    def __init__(
        self,
        X,
        df,
        session_name,
        n_perm=1000,
        metric="cosine",
        random_state=1234,
        null_model='shuffle_neurons', 
    ):
        """
        X : ndarray, shape (n_neurons, n_features)
            Readout weights (aligned with df rows)
        df : pandas.DataFrame
            Must contain column 'session_name'
        session_name : str
            Session to analyze
        n_perm : int
            Number of permutations
        null_model: str
            Two ways to generate null model:
                'shuffle_neurons': shuffle readout weights across neurons within the session
                'shuffle_weights': shuffle readout weights within each neuron
        """
        self.X = X
        self.df = df
        self.session_name = session_name
        self.n_perm = n_perm
        self.metric = metric
        self.rng = np.random.default_rng(random_state)
        self.null_model = null_model

        # Select session neurons
        self.idx = df["session_name"] == session_name
        self.Xs = X[self.idx.values]
        self.n = self.Xs.shape[0]

        if self.n < 2:
            raise ValueError("Need at least 2 neurons in session")

    def compute_observed(self):
        """Observed cosine distance matrix"""
        self.dist_obs_condensed = pdist(self.Xs, metric=self.metric) # shape (363 * 362 / 2, )
        self.dist_obs = squareform(self.dist_obs_condensed) # shape (363, 363)
        return self.dist_obs

    def compute_null(self):
        """Generate null distance matrices via permutation"""
        n_pairs = self.n * (self.n - 1) // 2
        null_dists_condensed = np.zeros((self.n_perm, n_pairs), dtype=np.float32)

        if self.null_model == 'shuffle_neurons':
            for k in range(self.n_perm):
                perm = self.rng.permutation(self.n)
                X_perm = self.Xs[perm]
                null_dists_condensed[k] = pdist(X_perm, metric=self.metric)
        elif self.null_model == 'shuffle_weights':
            for k in range(self.n_perm):
                Xp = np.zeros_like(self.Xs)
                for i in range(self.n):
                    Xp[i] = self.rng.permutation(self.Xs[i])
                null_dists_condensed[k] = pdist(Xp, metric=self.metric)

        self.null_dists_condensed = null_dists_condensed
        return null_dists_condensed

    def compute_pvalues(self):
        """Compute p-values for closeness to orthogonality (distance ≈ 1)."""
        if not hasattr(self, "dist_obs_condensed"):
            self.compute_observed()
        if not hasattr(self, "null_dists"):
            self.compute_null()

        # Distance-to-orthogonality statistic
        delta_obs = np.abs(self.dist_obs_condensed - 1.0)
        delta_null = np.abs(self.null_dists_condensed - 1.0)

        # One-sided permutation p-value: closer to 1 than null
        pvals = (
            1 + np.sum(delta_null <= delta_obs[None, :], axis=0)
        ) / (1 + self.n_perm)

        self.delta_obs_condensed = delta_obs
        self.pvals_condensed = pvals
        self.pvals = squareform(pvals)

        return self.pvals

    def significant_pairs(self, alpha=0.05):
        """Return indices of significantly orthogonal neuron pairs"""
        if not hasattr(self, "pvals"):
            self.compute_pvalues()

        i, j = np.triu_indices(self.n, k=1)
        mask = self.pvals[i, j] < alpha

        return np.vstack([i[mask], j[mask]]).T
    
    def significant_pairs_with_session_indices(self, alpha=0.05):
        """
        Return significant orthogonal neuron pairs with session-local indices.

        Returns
        -------
        pandas.DataFrame with columns:
            - session_neuron_idx_1
            - session_neuron_idx_2
            - p_value
            - cosine_distance
        """
        if not hasattr(self, "pvals"):
            self.compute_pvalues()

        # Local indices within the session
        i, j = np.triu_indices(self.n, k=1)
        mask = self.pvals[i, j] < alpha

        # Map local -> session neuron indices
        session_neuron_indices = self.df.loc[self.idx, "session_neuron_idx"].values

        data = {
            "session_neuron_idx_1": session_neuron_indices[i[mask]],
            "session_neuron_idx_2": session_neuron_indices[j[mask]],
            "p_value": self.pvals[i[mask], j[mask]],
            "cosine_distance": self.dist_obs[i[mask], j[mask]],
        }

        return pd.DataFrame(data)
    
    def plot_null_vs_observed(
        self,
        session_neuron_idx_1,
        session_neuron_idx_2,
        bins=40,
    ):
        """
        Plot null distribution and observed statistic for neuron-pair orthogonality.

        Orthogonality statistic:
            delta = |cosine_distance - 1|
        Smaller values indicate stronger orthogonality.
        """

        # Session-local neuron indices
        session_neuron_indices = self.df.loc[
            self.idx, "session_neuron_idx"
        ].values

        try:
            i = np.where(session_neuron_indices == session_neuron_idx_1)[0][0]
            j = np.where(session_neuron_indices == session_neuron_idx_2)[0][0]
        except IndexError:
            raise ValueError("Session neuron index not found")

        if i == j:
            raise ValueError("Neuron indices must be different")

        # Ensure (i, j) corresponds to condensed index ordering (upper triangle)
        if i > j:
            i, j = j, i

        # Condensed index mapping
        pair_idx = int(i * self.n - i * (i + 1) // 2 + (j - i - 1))

        # Observed statistics
        d_obs = self.dist_obs[i, j]
        delta_obs = abs(d_obs - 1.0)

        # Null statistics
        delta_null = abs(self.null_dists_condensed[:, pair_idx] - 1.0)

        # Correct permutation p-value
        p_val = (1 + np.sum(delta_null <= delta_obs)) / (1 + self.n_perm)

        # Plot
        plt.figure(figsize=(4, 3))
        plt.hist(delta_null, bins=bins, density=True, alpha=0.7, color='gray')
        plt.axvline(
            delta_obs,
            linestyle="--",
            linewidth=2,
            c='k',
            label=f"Observed |d−1| = {delta_obs:.4f}",
        )
        plt.xlabel("|Cosine distance − 1|")
        plt.ylabel("Density")
        plt.title(
            f"Orthogonality permutation test\n"
            f"{self.session_name}\n"
            f"Neurons {session_neuron_idx_1:.0f} vs {session_neuron_idx_2:.0f}: p = {p_val:.3f}"
        )
        plt.legend()
        plt.tight_layout()


def list_pad_concatenate(spike_list):
    """
    spike_list: a list of 2D or 3D arrays.
    
    - Concatenates along Axis 0 (Timepoints).
    - Pads Axis 1 (Locations) with NaN to match the largest found in the list.
    - Pads Axis 2 (Neurons), if it exists, to match the largest found in the list.
    """
    if not spike_list:
        return np.array([])
    # 1. Determine the dimensionality (2D or 3D)
    # We assume all arrays in the list have the same number of dimensions
    ndim = spike_list[0].ndim
    # 2. Find maximums for each dimension starting from axis 1
    # max_dims will store [max_axis_1, max_axis_2]
    max_dims = []
    for d in range(1, ndim):
        max_dims.append(max(arr.shape[d] for arr in spike_list))
    padded_list = []
    for arr in spike_list:
        # 3. Build a dynamic pad_width tuple
        # Axis 0: Always (0, 0) because we are concatenating here
        pad_config = [(0, 0)]
        # Axis 1 and (if exists) Axis 2
        for d in range(1, ndim):
            padding_needed = max_dims[d-1] - arr.shape[d]
            pad_config.append((0, padding_needed))
        # 4. Apply padding
        # astype(float) is required to support np.nan
        padded_arr = np.pad(
            arr.astype(float), 
            pad_width=pad_config, 
            mode='constant', 
            constant_values=np.nan
        )
        padded_list.append(padded_arr)
        
    # 5. Concatenate along the first dimension (Axis 0)
    return np.concatenate(padded_list, axis=0)


def compute_angles(vel):
    """
    vel: numpy array of shape (T, 3) representing [vx, vy, vz]
    Returns azimuth and elevation in degrees.
    """
    vx = vel[:, 0]
    vy = vel[:, 1]
    vz = vel[:, 2]

    # 1. Compute Azimuth (Yaw)
    # np.arctan2(y, x) handles the quadrant signs automatically
    azimuth = np.arctan2(vy, vx)

    # 2. Compute Elevation (Pitch)
    # We need the horizontal magnitude (projection on xy-plane)
    horizontal_mag = np.sqrt(vx**2 + vy**2)
    elevation = np.arctan2(vz, horizontal_mag)

    return np.degrees(azimuth), np.degrees(elevation)


def generate_landmark_dicts(xyz_dict, landmark_phy, threshold=0.3):
    reward_dict = {}
    perch_dict = {}
    for key, coords in xyz_dict.items():
        # 1. Identify which dataset ID this key belongs to (e.g., 'Dataset_1')
        dataset_id = None
        for ds_name in landmark_phy.keys():
            if ds_name in key:
                dataset_id = ds_name
                break
        if dataset_id is None:
            print(f"Warning: No matching dataset found in landmark_phy for key {key}")
            continue
        # Get the specific landmarks for this dataset
        landmarks = landmark_phy[dataset_id]  
        # 2. Helper function to find points within threshold of any landmark in a set
        def get_proximity_mask(data_pts, landmark_pts):
            # Use broadcasting: 
            # data_pts: (N, 1, 3) minus landmark_pts: (1, M, 3)
            # Resulting dists: (N, M)
            dists = np.linalg.norm(data_pts[:, np.newaxis, :] - landmark_pts[np.newaxis, :, :], axis=2)
            # Logical OR: Is the point within threshold of ANY landmark?
            is_near = np.any(dists < threshold, axis=1)
            return is_near
        # 3. Process Reward Locations
        rew_landmarks = landmarks['rew_locs']
        reward_dict[key] = get_proximity_mask(coords, rew_landmarks)
        # 4. Process Perch Locations
        perch_landmarks = landmarks['perch_locs']
        perch_dict[key] = get_proximity_mask(coords, perch_landmarks)
    return reward_dict, perch_dict


def extract_flight_type_saliency(
    all_saliency, 
    all_flight_session, 
    all_rel_time, 
    all_reward, 
    all_perch, 
    corr_dict, 
    corr_threshold=0.2,
):
    """
    Generate a dictionary: each key is a flight type such as ('Reward', 'Perch') and corresponding values are a list of 2D arrays, 
    each array with a shape of (number_of_timepoints_within_flight, number_of_neurons), representing neuronal responses at each flight trajectory. 
    And we also have a threshold 'corr_threshold' to use only neurons with predictive correlation values larger than the threshold.
    """
    # 1. Initialize the dictionary with flight types as keys
    flight_types = [
        ('Reward', 'Reward'), ('Reward', 'Perch'), ('Reward', 'Other'),
        ('Perch', 'Reward'), ('Perch', 'Perch'), ('Perch', 'Other'),
        ('Other', 'Reward'), ('Other', 'Perch'), ('Other', 'Other')
    ]
    result_dict = {ft: [] for ft in flight_types}
    # 2. Identify flight segments using relative time resets (0.0 to 1.0)
    # diff < 0 indicates the end of one flight and the start of the next
    jumps = np.where(np.diff(all_rel_time) < 0)[0]
    starts = np.concatenate(([0], jumps + 1))
    ends = np.concatenate((jumps, [len(all_rel_time) - 1]))

    for s, e in zip(starts, ends):
        # 3. Determine the Flight Type for this segment
        def get_loc_type(idx):
            if all_reward[idx]: return 'Reward'
            if all_perch[idx]: return 'Perch'
            return 'Other'
        current_ft = (get_loc_type(s), get_loc_type(e))
        
        # 4. Identify the Session Key
        # Format: 'Dataset_3_14543_240419_f1' -> Key: 'Dataset_3_14543_240419'
        full_flight_id = all_flight_session[s]
        session_key = "_".join(full_flight_id.split("_")[:-1])
        if session_key not in corr_dict:
            continue

        # 5. Filter Neurons by Correlation and Padding
        # Get correlations for this specific session
        session_corrs = corr_dict[session_key] # shape: (num_neurons_in_session,)
        
        # # Create a mask for neurons that:
        # # a) Exist in this session (not NaN in the spike data)
        # # b) Pass the correlation threshold
        # # Note: all_true_spike[s, :] is the row for the first timepoint of the flight
        # is_not_nan = ~np.isnan(all_true_spike[s, :len(session_corrs)])
        # is_above_threshold = session_corrs > corr_threshold
        # # Combined mask for the second dimension
        # # We handle potential size mismatch if session_corrs is shorter than 317
        # neuron_mask = np.zeros(all_true_spike.shape[1], dtype=bool)
        # valid_len = min(len(is_above_threshold), len(is_not_nan))
        # neuron_mask[:valid_len] = is_not_nan[:valid_len] & is_above_threshold[:valid_len]

        # Create a mask for neurons that passes the correlation threshold
        # because we know that num_neurons_in_session is not larger than all_true_spike.shape[-1]
        neuron_mask = np.where(session_corrs > corr_threshold)[0]

        # 6. Extract the slice and store
        # Shape: (timepoints_in_segment, timepoints_in_segment, selected_neurons)
        flight_response = all_saliency[s : e + 1, :e-s+1, neuron_mask]
        if flight_response.shape[-1] > 0:
            result_dict[current_ft].append(flight_response)

    return result_dict
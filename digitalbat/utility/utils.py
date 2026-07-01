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
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, wasserstein_distance
import ot
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats

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
    num_of_bins: int = None, # number of bins for the generate rate map in each spatial dimension
    dt: float = 0.1,  # Time interval between consecutive samples (in seconds)
    smooth_sigma: float = 1.0, # sigma for Gaussian smoothing
    rate_map_nan: bool = False, # If True, keep NaN values in rate map indicating the place never been visited
):
    """
    Compute the 2D rate map of a neuron by binning the 2D location space and calculating 
    the firing rate (in spikes per second) as spike count divided by time spent in each spatial bin.
    firing_rate_in_one_bin = total_spikes_in_one_bin / total_time_spent_in_one_bin

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

    # use either loc_bin_width or num_of_bins
    assert (loc_bin_width is None) != (num_of_bins is None), \
        "You must specify exactly one of 'loc_bin_width' or 'num_of_bins', not both or neither."
    # Define bin edges in x and y directions
    if loc_xx_lim is not None and loc_yy_lim is not None:
        if loc_bin_width:
            x_edges = np.arange(loc_xx_lim[0], loc_xx_lim[1] + loc_bin_width, loc_bin_width)
            y_edges = np.arange(loc_yy_lim[0], loc_yy_lim[1] + loc_bin_width, loc_bin_width)
        elif num_of_bins:
            x_edges = np.linspace(loc_xx_lim[0], loc_xx_lim[1], num_of_bins)
            y_edges = np.linspace(loc_yy_lim[0], loc_yy_lim[1], num_of_bins)
    else:
        if loc_bin_width:
            x_edges = np.arange(np.min(loc_xx), np.max(loc_xx) + loc_bin_width, loc_bin_width)
            y_edges = np.arange(np.min(loc_yy), np.max(loc_yy) + loc_bin_width, loc_bin_width)
        elif num_of_bins:
            x_edges = np.linspace(np.min(loc_xx), np.max(loc_xx), num_of_bins)
            y_edges = np.linspace(np.min(loc_yy), np.max(loc_yy), num_of_bins)

    # Compute occupancy: number of time points spent in each spatial bin
    occupancy, _, _ = np.histogram2d(loc_xx, loc_yy, bins=[x_edges, y_edges])
    # Convert occupancy count to time (in seconds)
    occupancy_time = occupancy * dt
    # Compute total spike count in each bin (weighted histogram)
    spike_map, _, _ = np.histogram2d(loc_xx, loc_yy, bins=[x_edges, y_edges], weights=unit_spike_count)

    if not rate_map_nan:
        # Compute firing rate: spike count / occupancy time
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_map = np.divide(spike_map, occupancy_time)
            rate_map[np.isnan(rate_map)] = 0  # Replace NaNs (from divide-by-zero) with 0
    else:
        occupancy_time[occupancy_time == 0] = np.nan # Avoid division by zero where the subject never visited
        rate_map = spike_map/occupancy_time # rate_map with NaN values
        occupancy_time[np.isnan(occupancy_time)] = 0

    # Smooth spike counts and occupancy separately, then divide.  Smoothing
    # the rate map directly leaks zero rates from unoccupied neighbor bins
    # into occupied ones and shrinks the time-averaged firing rate.
    # smooth_sigma = 1.0  # Standard deviation of Gaussian kernel (in units of spatial bins)
    spike_map_smoothed = gaussian_filter(spike_map, sigma=smooth_sigma)
    occupancy_time_smoothed = gaussian_filter(occupancy_time, sigma=smooth_sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed_rate_map = np.divide(spike_map_smoothed, occupancy_time_smoothed)
        smoothed_rate_map[np.isnan(smoothed_rate_map)] = 0

    return x_edges, y_edges, rate_map, smoothed_rate_map


def generate_rate_map(
    loc: np.ndarray, # the animal's location over time, shape: (T, 2) or (T, 3)
    unit_spike_count: np.ndarray,  # 1D array of spike counts for one neuron at each time point, shape: (T,)
    loc_lim: np.ndarray = None, # boundary (min + max) of the space, shape: (2, 3) or (2, 2)
    loc_bin_width: float = 0.1,  # Spatial bin width for both x and y dimensions (in location units)
    num_of_bins: int = None, # number of bins for the generate rate map in each spatial dimension
    dt: float = 0.1,  # Time interval between consecutive samples (in seconds)
    smooth_sigma: float = 1.0, # sigma for Gaussian smoothing
    rate_map_nan: bool = False, # If True, keep NaN values in rate map indicating the place never been visited
):
    # use either loc_bin_width or num_of_bins
    assert (loc_bin_width is None) != (num_of_bins is None), \
        "You must specify exactly one of 'loc_bin_width' or 'num_of_bins', not both or neither."
    
    if loc.shape[1] == 2: # 2D rate map
        # Define bin edges in x and y directions
        if loc_lim is not None:
            if loc_bin_width:
                x_edges = np.arange(loc_lim[0, 0], loc_lim[1, 0] + loc_bin_width, loc_bin_width)
                y_edges = np.arange(loc_lim[0, 1], loc_lim[1, 1] + loc_bin_width, loc_bin_width)
            elif num_of_bins:
                x_edges = np.linspace(loc_lim[0, 0], loc_lim[1, 0], num_of_bins)
                y_edges = np.linspace(loc_lim[0, 1], loc_lim[1, 1], num_of_bins)
        else:
            if loc_bin_width:
                x_edges = np.arange(np.min(loc[:, 0]), np.max(loc[:, 0]) + loc_bin_width, loc_bin_width)
                y_edges = np.arange(np.min(loc[:, 1]), np.max(loc[:, 1]) + loc_bin_width, loc_bin_width)
            elif num_of_bins:
                x_edges = np.linspace(np.min(loc[:, 0]), np.max(loc[:, 0]), num_of_bins)
                y_edges = np.linspace(np.min(loc[:, 1]), np.max(loc[:, 1]), num_of_bins)
        # Compute occupancy: number of time points spent in each spatial bin
        occupancy, _, _ = np.histogram2d(loc[:, 0], loc[:, 1], bins=[x_edges, y_edges])
        # Convert occupancy count to time (in seconds)
        occupancy_time = occupancy * dt
        # Compute total spike count in each bin (weighted histogram)
        spike_map, _, _ = np.histogram2d(loc[:, 0], loc[:, 1], bins=[x_edges, y_edges], weights=unit_spike_count)

    elif loc.shape[1] ==3: # 3D rate map
        # Define bin edges in x and y directions
        if loc_lim is not None:
            if loc_bin_width:
                x_edges = np.arange(loc_lim[0, 0], loc_lim[1, 0] + loc_bin_width, loc_bin_width)
                y_edges = np.arange(loc_lim[0, 1], loc_lim[1, 1] + loc_bin_width, loc_bin_width)
                z_edges = np.arange(loc_lim[0, 2], loc_lim[1, 2] + loc_bin_width, loc_bin_width)
            elif num_of_bins:
                x_edges = np.linspace(loc_lim[0, 0], loc_lim[1, 0], num_of_bins)
                y_edges = np.linspace(loc_lim[0, 1], loc_lim[1, 1], num_of_bins)
                z_edges = np.linspace(loc_lim[0, 2], loc_lim[1, 2], num_of_bins)
        else:
            if loc_bin_width:
                x_edges = np.arange(np.min(loc[:, 0]), np.max(loc[:, 0]) + loc_bin_width, loc_bin_width)
                y_edges = np.arange(np.min(loc[:, 1]), np.max(loc[:, 1]) + loc_bin_width, loc_bin_width)
                z_edges = np.arange(np.min(loc[:, 2]), np.max(loc[:, 2]) + loc_bin_width, loc_bin_width)
            elif num_of_bins:
                x_edges = np.linspace(np.min(loc[:, 0]), np.max(loc[:, 0]), num_of_bins)
                y_edges = np.linspace(np.min(loc[:, 1]), np.max(loc[:, 1]), num_of_bins)
                z_edges = np.linspace(np.min(loc[:, 2]), np.max(loc[:, 2]), num_of_bins)
        # Compute occupancy: number of time points spent in each spatial bin
        occupancy, _ = np.histogramdd(loc, bins=[x_edges, y_edges, z_edges])
        # Convert occupancy count to time (in seconds)
        occupancy_time = occupancy * dt
        # Compute total spike count in each bin (weighted histogram)
        spike_map, _ = np.histogramdd(loc, bins=[x_edges, y_edges, z_edges], weights=unit_spike_count)

    if not rate_map_nan:
        # Compute firing rate: spike count / occupancy time
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_map = np.divide(spike_map, occupancy_time)
            rate_map[np.isnan(rate_map)] = 0  # Replace NaNs (from divide-by-zero) with 0
    else:
        occupancy_time[occupancy_time == 0] = np.nan # Avoid division by zero where the subject never visited
        rate_map = spike_map/occupancy_time # rate_map with NaN values
        occupancy_time[np.isnan(occupancy_time)] = 0

    # Smooth spike counts and occupancy separately, then divide.  Smoothing
    # the rate map directly leaks zero rates from unoccupied neighbor bins
    # into occupied ones and shrinks the time-averaged firing rate.
    # smooth_sigma = 1.0  # Standard deviation of Gaussian kernel (in units of spatial bins)
    spike_map_smoothed = gaussian_filter(spike_map, sigma=smooth_sigma)
    occupancy_time_smoothed = gaussian_filter(occupancy_time, sigma=smooth_sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed_rate_map = np.divide(spike_map_smoothed, occupancy_time_smoothed)
        smoothed_rate_map[np.isnan(smoothed_rate_map)] = 0

    return ((x_edges, y_edges), rate_map, smoothed_rate_map) if loc.shape[1] == 2 else ((x_edges, y_edges, z_edges), rate_map, smoothed_rate_map)


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


class BatSpatialInfo:
    """
    For a target session, compute spatial_info in the physical space and the 3D latent space, the coordinates in
    the physical space and the 3D latent space, spikes for each neuron and each flight function type (such as reward -> reward)
    and each flight geometry cluster in a flight function type.

    all_xyz.shape: (num_of_timepoints, 3)
    all_true_spike.shape: (num_of_timepoints, num_of_neurons)
    """
    def __init__(self, session_id, all_session, all_xyz, pca_sub, all_latents, 
                 all_true_spike, all_cluster_session, all_rel_time, all_reward, all_perch, dt=0.1):
        self.session_id = session_id
        self.dt = dt
        # Mask for the specific session
        self.session_mask = (all_session == session_id)
        # Extract session-specific data
        self.xyz = all_xyz[self.session_mask]
        self.pca_3d = pca_sub[self.session_mask]
        self.latent_32d = all_latents[self.session_mask]
        self.clusters = all_cluster_session[self.session_mask]
        # Extract session-specific behavioral variables
        session_rel_time = all_rel_time[self.session_mask]
        session_reward = all_reward[self.session_mask]
        session_perch = all_perch[self.session_mask]
        # Get spikes and drop neurons that are all NaN for this session
        session_spikes = all_true_spike[self.session_mask]
        self.valid_neuron_mask = ~np.all(np.isnan(session_spikes), axis=0)
        self.spikes = session_spikes[:, self.valid_neuron_mask] # Shape: (T_session, N_valid)
        self.n_time, self.n_neurons = self.spikes.shape
        # --- Identify Flight Functional Types per time point ---
        self.flight_types = np.empty(self.n_time, dtype=object)
        if self.n_time > 0:
            # Find the indices where a new flight begins (relative time drops back to 0)
            jumps = np.where(np.diff(session_rel_time) < 0)[0]
            # Flight starts at index 0 and every index right after a jump
            starts = np.concatenate(([0], jumps + 1))
            # Flight ends at each jump index, and the final flight ends at the last index of the session
            ends = np.concatenate((jumps, [self.n_time - 1]))
            # Assign functional type to every timepoint within each flight segment
            for s, e in zip(starts, ends):
                # Guard rails to make sure indices don't overflow or create empty slices
                if s >= self.n_time or e >= self.n_time or s > e:
                    continue
                s_type = 'Reward' if session_reward[s] else ('Perch' if session_perch[s] else 'Other')
                e_type = 'Reward' if session_reward[e] else ('Perch' if session_perch[e] else 'Other')
                if s_type != 'Other' and e_type != 'Other':
                    ft_key = (s_type, e_type)
                else:
                    ft_key = ('Other',)
                self.flight_types[s:e+1] = str(ft_key)

    def _get_equal_occ_bins(self, coords, n_bins_per_dim=10):
        """Calculates bin indices for each point based on coordinate ranges."""
        bin_indices = np.zeros_like(coords, dtype=int)
        for d in range(3): 
            edges = np.linspace(np.min(coords[:, d]), np.max(coords[:, d]), n_bins_per_dim + 1)
            bin_indices[:, d] = np.digitize(coords[:, d], edges[1:-1])
        return bin_indices[:, 0] + \
               n_bins_per_dim * bin_indices[:, 1] + \
               (n_bins_per_dim**2) * bin_indices[:, 2]

    def compute_skaggs_si(self, coords, spikes, n_bins_per_dim=20):
        """Computes Skaggs Spatial Information (bits/spike)."""
        bin_ids = self._get_equal_occ_bins(coords, n_bins_per_dim)
        n_total_bins = n_bins_per_dim**3
        occ_counts = np.bincount(bin_ids, minlength=n_total_bins)
        Pi = occ_counts / len(bin_ids)
        si_per_neuron = []
        for n in range(spikes.shape[1]):
            r_vec = spikes[:, n]
            mean_rate = np.mean(r_vec)
            if mean_rate == 0:
                si_per_neuron.append(0.0)
                continue
            bin_sums = np.bincount(bin_ids, weights=r_vec, minlength=n_total_bins)
            lambda_i = np.divide(bin_sums, occ_counts, out=np.zeros_like(bin_sums), where=occ_counts > 0)
            ratio = lambda_i / mean_rate
            mask = (ratio > 0) & (Pi > 0)
            info = np.sum(Pi[mask] * ratio[mask] * np.log2(ratio[mask]))
            si_per_neuron.append(info)
        return np.array(si_per_neuron)

    def compute_ksg_si(self, high_dim_coords, spikes):
        """High-dimensional SI using KSG estimator (Mutual Information)."""
        mi_scores = []
        for n in range(spikes.shape[1]):
            score = mutual_info_regression(high_dim_coords, spikes[:, n], 
                                           discrete_features=False, n_neighbors=3)
            mi_scores.append(score[0] / np.log(2))
        return np.array(mi_scores)

    def get_all_spatial_info(self):
        spatial_info = {}
        # 1. Session SI (3D Physical vs 3D Latent)
        spatial_info['session_si_physical'] = self.compute_skaggs_si(self.xyz, self.spikes)
        spatial_info['session_si_latent_3d'] = self.compute_skaggs_si(self.pca_3d, self.spikes)
        # 2. Cluster-specific SI
        unique_clusters = np.unique(self.clusters)
        spatial_info['clusters'] = {}
        for cluster in unique_clusters:
            c_mask = (self.clusters == cluster)
            if np.sum(c_mask) < 10: continue 
            spatial_info['clusters'][cluster] = {
                'si_physical': self.compute_skaggs_si(self.xyz[c_mask], self.spikes[c_mask]),
                'si_latent_3d': self.compute_skaggs_si(self.pca_3d[c_mask], self.spikes[c_mask]),
                'xyz': self.xyz[c_mask],
                'pca_3d': self.pca_3d[c_mask],
                'spikes': self.spikes[c_mask],
            }
            
        # 3. Hierarchical Hierachy: Flight Function Types -> Clusters
        spatial_info['flight_function_types'] = {}
        # Defined targets
        target_types = [
            ('Reward', 'Reward'), ('Reward', 'Perch'), ('Perch', 'Reward'), ('Perch', 'Perch'), ('Other',)
        ]
        for ft in target_types:
            ft_str = str(ft)
            ft_mask = (self.flight_types == ft_str)
            if np.sum(ft_mask) < 10: continue
            spatial_info['flight_function_types'][ft] = {'clusters': {}}
            # Sub-sample clusters within this specific flight function block
            ft_clusters = self.clusters[ft_mask]
            unique_ft_clusters = np.unique(ft_clusters)
            for cluster in unique_ft_clusters:
                # Combined mask: must belong to the functional type AND the geometry cluster
                combined_mask = (self.flight_types == ft_str) & (self.clusters == cluster)
                if np.sum(combined_mask) < 10: continue
                spatial_info['flight_function_types'][ft]['clusters'][cluster] = {
                    'si_physical': self.compute_skaggs_si(self.xyz[combined_mask], self.spikes[combined_mask]),
                    'si_latent_3d': self.compute_skaggs_si(self.pca_3d[combined_mask], self.spikes[combined_mask]),
                    'xyz': self.xyz[combined_mask],
                    'pca_3d': self.pca_3d[combined_mask],
                    'spikes': self.spikes[combined_mask],
                }
        # 4. High-Dimensional SI (32D Latent)
        spatial_info['global_si_latent_32d'] = self.compute_ksg_si(self.latent_32d, self.spikes)
        
        return spatial_info
    
    def compute_shuffled_si(self, neuron_idx, cluster_id=None, n_shuffles=1000):
        """Calculates null distribution by circularly shifting spike trains."""
        if cluster_id:
            c_mask = (self.clusters == cluster_id)
            coords = self.pca_3d[c_mask] # or self.xyz, self.pca_3d
            spikes = self.spikes[c_mask, neuron_idx]
        else:
            coords = self.pca_3d
            spikes = self.spikes[:, neuron_idx]
            
        n_samples = len(spikes)
        null_distribution = []
        # Pre-calculate bin IDs for original coordinates
        bin_ids = self._get_equal_occ_bins(coords)
        for _ in range(n_shuffles):
            # Circular shift: maintain temporal structure but decouple from position
            shift = np.random.randint(int(0.1*n_samples), int(0.9*n_samples))
            shuffled_spikes = np.roll(spikes, shift)
            # Calculate SI for this shuffle
            mean_rate = np.mean(shuffled_spikes)
            if mean_rate == 0:
                null_distribution.append(0.0)
                continue
                
            occ_counts = np.bincount(bin_ids, minlength=1000)
            bin_sums = np.bincount(bin_ids, weights=shuffled_spikes, minlength=1000)
            lambda_i = np.divide(bin_sums, occ_counts, out=np.zeros_like(bin_sums), where=occ_counts > 0)
            
            Pi = occ_counts / n_samples
            ratio = lambda_i / mean_rate
            mask = (ratio > 0) & (Pi > 0)
            null_distribution.append(np.sum(Pi[mask] * ratio[mask] * np.log2(ratio[mask])))
            
        return np.array(null_distribution)
    

class BatRateMapSim:
    """
    For a target session, compute the similarities between rate maps in the physical space and the 3D latent space, 
    using different metrics, for each neuron and each flight function type (such as reward -> reward)
    and each flight geometry cluster in a flight function type.

    all_xyz.shape: (num_of_timepoints, 3)
    all_true_spike.shape: (num_of_timepoints, num_of_neurons)
    """
    def __init__(self, session_id, all_session, all_xyz, pca_sub, all_latents, 
                 all_true_spike, all_cluster_session, all_rel_time, all_reward, all_perch, dt=0.1):
        self.session_id = session_id
        self.dt = dt
        # Mask for the specific session
        self.session_mask = (all_session == session_id)
        # Extract session-specific coordinate limits
        self.xyz = all_xyz[self.session_mask]
        self.pca_3d = pca_sub[self.session_mask]
        self.latent_32d = all_latents[self.session_mask]
        self.clusters = all_cluster_session[self.session_mask]
        # Global session boundary limits shared across all internal clusters
        self.loc_xyz_lim = np.stack((np.min(self.xyz, axis=0), np.max(self.xyz, axis=0)), axis=0) # shape: (2, 3)
        self.loc_pca_lim = np.stack((np.min(self.pca_3d, axis=0), np.max(self.pca_3d, axis=0)), axis=0)
        # Extract behavioral variables
        session_rel_time = all_rel_time[self.session_mask]
        session_reward = all_reward[self.session_mask]
        session_perch = all_perch[self.session_mask]
        # Spikes array matching valid neurons
        session_spikes = all_true_spike[self.session_mask]
        self.valid_neuron_mask = ~np.all(np.isnan(session_spikes), axis=0)
        self.spikes = session_spikes[:, self.valid_neuron_mask]
        self.n_time, self.n_neurons = self.spikes.shape
        # Identify Flight Functional Types per time point
        self.flight_types = np.empty(self.n_time, dtype=object)
        if self.n_time > 0:
            jumps = np.where(np.diff(session_rel_time) < 0)[0]
            starts = np.concatenate(([0], jumps + 1))
            ends = np.concatenate((jumps, [self.n_time - 1]))
            for s, e in zip(starts, ends):
                if s >= self.n_time or e >= self.n_time or s > e:
                    continue
                s_type = 'Reward' if session_reward[s] else ('Perch' if session_perch[s] else 'Other')
                e_type = 'Reward' if session_reward[e] else ('Perch' if session_perch[e] else 'Other')
                ft_key = (s_type, e_type) if s_type != 'Other' and e_type != 'Other' else ('Other',)
                self.flight_types[s:e+1] = str(ft_key)

    def _compute_pairwise_similarity(self, maps_list, metric_func):
        """Helper to calculate mean similarity of all unique pairs of maps for each neuron."""
        n_clusters = len(maps_list)
        if n_clusters <= 1:
            return np.full(self.n_neurons, np.nan)
        # Accumulator array shape: (num_pairs, n_neurons)
        pair_scores = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                map_a = maps_list[i] # Shape: (num_bins, num_bins, num_bins, n_neurons)
                map_b = maps_list[j]
                scores_per_neuron = np.zeros(self.n_neurons)
                for n in range(self.n_neurons):
                    scores_per_neuron[n] = metric_func(map_a[..., n], map_b[..., n])
                pair_scores.append(scores_per_neuron)
                
        return np.mean(np.array(pair_scores), axis=0)

    def _get_valid_cluster_maps(self, ft_str, space_type='physical', num_of_bins=20):
        """Generates smoothed 3D rate maps for clusters within a functional type.
        
        If a neuron has no spikes within a cluster, its rate map is set to all NaNs.
        """
        ft_mask = (self.flight_types == ft_str)
        unique_clusters = np.unique(self.clusters[ft_mask])
        coords = self.xyz if space_type == 'physical' else self.pca_3d
        lims = self.loc_xyz_lim if space_type == 'physical' else self.loc_pca_lim
        
        maps_list = []
        for clu in unique_clusters:
            combined_mask = (self.flight_types == ft_str) & (self.clusters == clu)
            if np.sum(combined_mask) < 10:
                continue
            # Pre-allocate array: (Bins, Bins, Bins, Neurons)
            cluster_maps = np.zeros((num_of_bins - 1, num_of_bins - 1, num_of_bins - 1, self.n_neurons))
            for n in range(self.n_neurons):
                # Check if the neuron actually spikes in this specific cluster segment
                if np.sum(self.spikes[combined_mask, n]) == 0:
                    cluster_maps[..., n] = np.nan  # Mark this neuron as inactive for this cluster
                    continue
                # Generate 3D maps using smoothed_rate_map (index 2)
                _, _, r_map = generate_rate_map(
                    loc=coords[combined_mask],
                    unit_spike_count=self.spikes[combined_mask, n],
                    loc_lim=lims,
                    loc_bin_width=None,
                    num_of_bins=num_of_bins,
                    dt=self.dt
                )
                cluster_maps[..., n] = r_map
                
            maps_list.append(cluster_maps)
        return maps_list

    def get_all_corr(self):
        """Computes average spatial Pearson correlations for visited bins."""
        spatial_corr = {'flight_function_types': {}}
        target_types = [('Reward', 'Reward'), ('Reward', 'Perch'), ('Perch', 'Reward'), ('Perch', 'Perch'), ('Other',)]
        
        def pearson_metric(m1, m2):
            # If either map belongs to an inactive neuron for this cluster, return NaN
            if np.any(np.isnan(m1)) or np.any(np.isnan(m2)):
                return np.nan
            visited = (m1 > 0) & (m2 > 0)
            if np.sum(visited) < 5 or np.std(m1[visited]) == 0 or np.std(m2[visited]) == 0:
                return np.nan
            return pearsonr(m1[visited], m2[visited])[0]

        for ft in target_types:
            ft_str = str(ft)
            if np.sum(self.flight_types == ft_str) == 0: continue
            
            phys_maps = self._get_valid_cluster_maps(ft_str, 'physical')
            lat_maps = self._get_valid_cluster_maps(ft_str, 'latent')
            
            spatial_corr['flight_function_types'][ft] = {
                'corr_physical': self._compute_pairwise_similarity(phys_maps, pearson_metric),
                'corr_latent': self._compute_pairwise_similarity(lat_maps, pearson_metric)
            }
        return spatial_corr

    def get_all_Wasserstein(self, num_of_bins=20):
        """Computes Earth Mover's Distance similarity profiles."""
        spatial_wass = {'flight_function_types': {}}
        target_types = [('Reward', 'Reward'), ('Reward', 'Perch'), ('Perch', 'Reward'), ('Perch', 'Perch'), ('Other',)]
        
        # def wasserstein_metric(m1, m2):
        #     if np.any(np.isnan(m1)) or np.any(np.isnan(m2)):
        #         return np.nan
        #     v1, v2 = m1.flatten(), m2.flatten()
        #     if np.sum(v1) == 0 or np.sum(v2) == 0:
        #         return np.nan
        #     return wasserstein_distance(v1 / np.sum(v1), v2 / np.sum(v2))
        
        # Pre-compute the 3D grid coordinate locations once globally 
        # so we don't recalculate them inside the heavy nested loops
        grid_coords = np.array(np.meshgrid(np.arange(num_of_bins - 1), np.arange(num_of_bins - 1), np.arange(num_of_bins - 1), indexing='ij')).reshape(3, -1).T
        # Compute the pairwise Euclidean distance matrix between all 8000 bins (20x20x20)
        # This represents the actual "metric distance space" for the transport cost
        M_cost = ot.dist(grid_coords, grid_coords, metric='euclidean')
        def wasserstein_metric(m1, m2):
            # If either map belongs to an inactive neuron for this cluster, return NaN
            if np.any(np.isnan(m1)) or np.any(np.isnan(m2)):
                return np.nan
            v1, v2 = m1.flatten(), m2.flatten()
            # Sum of weights must be > 0 to normalize
            if np.sum(v1) == 0 or np.sum(v2) == 0:
                return np.nan
            # EMD requires the mass distributions to sum to 1 (probability distributions)
            p1 = v1 / np.sum(v1)
            p2 = v2 / np.sum(v2)
            # Compute the true multi-dimensional Earth Mover's Distance
            # ot.emd2 returns the total scalar cost: sum(mass_moved * Euclidean_distance)
            return ot.emd2(p1, p2, M_cost)

        for ft in target_types:
            ft_str = str(ft)
            if np.sum(self.flight_types == ft_str) == 0: continue
            
            phys_maps = self._get_valid_cluster_maps(ft_str, 'physical')
            lat_maps = self._get_valid_cluster_maps(ft_str, 'latent')
            
            spatial_wass['flight_function_types'][ft] = {
                'wass_physical': self._compute_pairwise_similarity(phys_maps, wasserstein_metric),
                'wass_latent': self._compute_pairwise_similarity(lat_maps, wasserstein_metric)
            }
        return spatial_wass

    def get_all_Jaccard(self):
        """Computes Jaccard Intersection-over-Union similarity on thresholded place fields."""
        spatial_jac = {'flight_function_types': {}}
        target_types = [('Reward', 'Reward'), ('Reward', 'Perch'), ('Perch', 'Reward'), ('Perch', 'Perch'), ('Other',)]
        
        def jaccard_metric(m1, m2):
            if np.any(np.isnan(m1)) or np.any(np.isnan(m2)):
                return np.nan
            field1 = m1 > (0.2 * np.max(m1)) if np.max(m1) > 0 else np.zeros_like(m1, dtype=bool)
            field2 = m2 > (0.2 * np.max(m2)) if np.max(m2) > 0 else np.zeros_like(m2, dtype=bool)
            
            intersection = np.sum(field1 & field2)
            union = np.sum(field1 | field2)
            return intersection / union if union > 0 else np.nan

        for ft in target_types:
            ft_str = str(ft)
            if np.sum(self.flight_types == ft_str) == 0: continue
            
            phys_maps = self._get_valid_cluster_maps(ft_str, 'physical')
            lat_maps = self._get_valid_cluster_maps(ft_str, 'latent')
            
            spatial_jac['flight_function_types'][ft] = {
                'jaccard_physical': self._compute_pairwise_similarity(phys_maps, jaccard_metric),
                'jaccard_latent': self._compute_pairwise_similarity(lat_maps, jaccard_metric)
            }
        return spatial_jac
    
    def get_shuffled_jaccard_baseline(self, ft_str, n_shuffles=20, num_of_bins=20):
        """
        Computes a shuffled baseline for Jaccard indices by circularly shifting 
        the spike trains before rate map generation. This preserves the exact
        behavioral occupancy distributions,
        but destroys the spatial relationship to neural tuning.
        """
        ft_mask = (self.flight_types == ft_str)
        unique_clusters = np.unique(self.clusters[ft_mask])
        
        # Target types to evaluate side-by-side
        spaces = ['physical', 'latent']
        shuffled_results = {'physical': [], 'latent': []}
        
        for space in spaces:
            coords = self.xyz if space == 'physical' else self.pca_3d
            lims = self.loc_xyz_lim if space == 'physical' else self.loc_pca_lim
            
            # We will collect pairwise similarities across multiple random shuffles
            shuffle_pairs = []
            
            for shuffle_idx in range(n_shuffles):
                maps_list = []
                
                for clu in unique_clusters:
                    combined_mask = (self.flight_types == ft_str) & (self.clusters == clu)
                    if np.sum(combined_mask) < 10:
                        continue
                    
                    cluster_maps = np.zeros((num_of_bins - 1, num_of_bins - 1, num_of_bins - 1, self.n_neurons))
                    
                    for n in range(self.n_neurons):
                        if np.sum(self.spikes[combined_mask, n]) == 0:
                            cluster_maps[..., n] = np.nan
                            continue
                        
                        # --- THE SHUFFLE ---
                        # Circularly shift spikes by a random offset to break spatial correlation 
                        # while keeping the temporal structure of the spikes and the exact occupancy intact.
                        raw_spikes = self.spikes[combined_mask, n]
                        shift_offset = np.random.randint(int(len(raw_spikes) * 0.2), int(len(raw_spikes) * 0.8))
                        shuffled_spikes = np.roll(raw_spikes, shift_offset)
                        
                        # Generate rate map with intact occupancy but shuffled spikes
                        _, _, r_map = generate_rate_map(
                            loc=coords[combined_mask],
                            unit_spike_count=shuffled_spikes,
                            loc_lim=lims,
                            loc_bin_width=None,
                            num_of_bins=num_of_bins,
                            dt=self.dt
                        )
                        cluster_maps[..., n] = r_map
                        
                    maps_list.append(cluster_maps)
                
                # Compute pairwise similarities for this shuffle iteration
                if len(maps_list) > 1:
                    # Re-using your exact Jaccard metric logic
                    def jaccard_metric(m1, m2):
                        if np.any(np.isnan(m1)) or np.any(np.isnan(m2)): return np.nan
                        f1 = m1 > (0.2 * np.max(m1)) if np.max(m1) > 0 else np.zeros_like(m1, dtype=bool)
                        f2 = m2 > (0.2 * np.max(m2)) if np.max(m2) > 0 else np.zeros_like(m2, dtype=bool)
                        union = np.sum(f1 | f2)
                        return np.sum(f1 & f2) / union if union > 0 else np.nan
                    
                    # Compute average across pairs for this shuffle
                    mean_scores = self._compute_pairwise_similarity(maps_list, jaccard_metric)
                    shuffle_pairs.append(mean_scores)
                    
            if len(shuffle_pairs) > 0:
                # Average across shuffles to get a stable baseline per neuron
                shuffled_results[space] = np.nanmean(np.array(shuffle_pairs), axis=0)
                
        return shuffled_results
    
    def get_shuffled_wass_baseline(self, ft_str, n_shuffles=20, num_of_bins=20):
        """
        Computes a shuffled baseline for Wasserstein distances by circularly shifting 
        the spike trains before rate map generation. This preserves the exact
        behavioral occupancy distributions,
        but destroys the spatial relationship to neural tuning.
        """
        ft_mask = (self.flight_types == ft_str)
        unique_clusters = np.unique(self.clusters[ft_mask])
        
        # Target types to evaluate side-by-side
        spaces = ['physical', 'latent']
        shuffled_results = {'physical': [], 'latent': []}
        
        for space in spaces:
            coords = self.xyz if space == 'physical' else self.pca_3d
            lims = self.loc_xyz_lim if space == 'physical' else self.loc_pca_lim
            
            # We will collect pairwise similarities across multiple random shuffles
            shuffle_pairs = []
            
            for shuffle_idx in range(n_shuffles):
                print(f'shuffle_idx: {shuffle_idx}')
                maps_list = []
                
                for clu in unique_clusters:
                    combined_mask = (self.flight_types == ft_str) & (self.clusters == clu)
                    if np.sum(combined_mask) < 10:
                        continue
                    
                    cluster_maps = np.zeros((num_of_bins - 1, num_of_bins - 1, num_of_bins - 1, self.n_neurons))
                    
                    for n in range(self.n_neurons):
                        if np.sum(self.spikes[combined_mask, n]) == 0:
                            cluster_maps[..., n] = np.nan
                            continue
                        
                        # --- THE SHUFFLE ---
                        # Circularly shift spikes by a random offset to break spatial correlation 
                        # while keeping the temporal structure of the spikes and the exact occupancy intact.
                        raw_spikes = self.spikes[combined_mask, n]
                        shift_offset = np.random.randint(int(len(raw_spikes) * 0.2), int(len(raw_spikes) * 0.8))
                        shuffled_spikes = np.roll(raw_spikes, shift_offset)
                        
                        # Generate rate map with intact occupancy but shuffled spikes
                        _, _, r_map = generate_rate_map(
                            loc=coords[combined_mask],
                            unit_spike_count=shuffled_spikes,
                            loc_lim=lims,
                            loc_bin_width=None,
                            num_of_bins=num_of_bins,
                            dt=self.dt
                        )
                        cluster_maps[..., n] = r_map
                        
                    maps_list.append(cluster_maps)
                
                # Compute pairwise similarities for this shuffle iteration
                if len(maps_list) > 1:
                    # Re-using your exact Wasserstein distane logic
                    grid_coords = np.array(np.meshgrid(np.arange(num_of_bins - 1), np.arange(num_of_bins - 1), np.arange(num_of_bins - 1), indexing='ij')).reshape(3, -1).T
                    M_cost = ot.dist(grid_coords, grid_coords, metric='euclidean')
                    def wasserstein_metric(m1, m2):
                        if np.any(np.isnan(m1)) or np.any(np.isnan(m2)):
                            return np.nan
                        v1, v2 = m1.flatten(), m2.flatten()
                        if np.sum(v1) == 0 or np.sum(v2) == 0:
                            return np.nan
                        p1 = v1 / np.sum(v1)
                        p2 = v2 / np.sum(v2)
                        return ot.emd2(p1, p2, M_cost)
                    
                    # Compute average across pairs for this shuffle
                    mean_scores = self._compute_pairwise_similarity(maps_list, wasserstein_metric)
                    shuffle_pairs.append(mean_scores)
                    
            if len(shuffle_pairs) > 0:
                # Average across shuffles to get a stable baseline per neuron
                shuffled_results[space] = np.nanmean(np.array(shuffle_pairs), axis=0)
                
        return shuffled_results
    


def decode_spikes_to_latent(all_true_spike, all_latents, all_session, alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]):
    """
    Projects raw session-wise spikes into the latent space using a cross-validated 
    Ridge regression encoder with a 4-1-1 (train-val-test) split across 6 folds.
    
    Parameters:
    -----------
    all_true_spike : np.ndarray
        Shape (8332, 266), neural spike matrix containing NaN paddings.
    all_latents : np.ndarray
        Shape (8332, 32), true latent representations.
    all_session : np.ndarray
        Shape (8332,), 1D array of session names matching each time point.
    alphas : list
        List of L2 regularization hyperparameters to tune.
        
    Returns:
    --------
    all_spike_in_latent : np.ndarray
        Shape (8332, 32), out-of-sample reconstructed latents aligned with original order.
    """
    # Initialize the final output array matching the shape of all_latents
    all_spike_in_latent = np.full_like(all_latents, np.nan)
    unique_sessions = np.unique(all_session)
    print(f"Starting latent reconstruction across {len(unique_sessions)} sessions...\n")
    for sess in unique_sessions:
        # Step 1: Extract session data and keep position masks
        sess_mask = (all_session == sess)
        X_sess_raw = all_true_spike[sess_mask] # Spikes for this session
        Y_sess = all_latents[sess_mask]        # Target latents for this session
        # Remove columns (neurons) that are entirely NaN values in this session
        valid_neurons_mask = ~np.isnan(X_sess_raw).all(axis=0)
        X_sess = X_sess_raw[:, valid_neurons_mask]
        
        n_samples = X_sess.shape[0]
        # Guard rail: ensure we have enough data points to do a 6-fold split
        if n_samples < 6:
            print(f"Skipping session {sess}: too few samples ({n_samples}).")
            continue
        
        # PREPROCESSING BLOCK (SESSION-SPECIFIC)
        # A. Transform Spikes: Log(x+1) or Sqrt(x) to stabilize Poisson variance
        # We'll use Log1p here as it is robust to high-count bursting transients
        # X_sess = np.log1p(X_sess)
        # B. Z-Score Spikes (Scale features symmetrically for Ridge L2 penalty)
        scaler_X = StandardScaler()
        X_sess = scaler_X.fit_transform(X_sess)
        # C. Z-Score Latents (Ensures multi-output R2 treats all 32 dims equally)
        scaler_Y = StandardScaler()
        Y_sess = scaler_Y.fit_transform(Y_sess)

        # Initialize an array to hold out-of-sample predictions for this specific session
        sess_predictions = np.full_like(Y_sess, np.nan)
        # Steps 2 & 3: 6-Fold Cross Validation with Native Shuffling
        # Setting random_state ensures reproducibility while shuffling
        kf = KFold(n_splits=6, shuffle=True, random_state=42)
        folds = list(kf.split(X_sess))
        # Track R2 per fold to log encoding performance
        fold_r2_scores = []
        for test_fold_idx in range(6):
            # Extract test indices
            _, test_idx = folds[test_fold_idx]
            # Select the next fold in line as the validation fold (wrap around using modulo)
            val_fold_idx = (test_fold_idx + 1) % 6
            _, val_idx = folds[val_fold_idx]
            # The remaining 4 folds are designated for training
            train_indices_list = [folds[i][1] for i in range(6) if i != test_fold_idx and i != val_fold_idx]
            train_idx = np.concatenate(train_indices_list)
            # Split data for this fold configuration
            X_train, Y_train = X_sess[train_idx], Y_sess[train_idx]
            X_val, Y_val = X_sess[val_idx], Y_sess[val_idx]
            X_test, _ = X_sess[test_idx], Y_sess[test_idx]
            # --- Hyperparameter Tuning (L2 Alpha) on Validation Fold ---
            best_alpha = alphas[0]
            best_val_r2 = -np.inf
            for alpha in alphas:
                # Fit model on 4 folds
                model = Ridge(alpha=alpha)
                model.fit(X_train, Y_train)
                # Evaluate on 1 validation fold
                val_pred = model.predict(X_val)
                val_r2 = r2_score(Y_val, val_pred, multioutput='variance_weighted')
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_alpha = alpha
            
            # --- Final Evaluation on Test Fold ---
            # Retrain model using the optimized alpha on training data
            final_model = Ridge(alpha=best_alpha)
            final_model.fit(X_train, Y_train)
            # Predict the out-of-sample test fold
            test_pred = final_model.predict(X_test)
            # Step 4: Write directly back to original session position indices
            # Because kf.split gives indices relative to X_sess, this maps directly back 
            # to the native temporal structure without manual un-shuffling.
            sess_predictions[test_idx] = test_pred
            
            # Calculate testing R2 score for tracking
            test_r2 = r2_score(Y_sess[test_idx], test_pred, multioutput='variance_weighted')
            fold_r2_scores.append(test_r2)
            
        # Step 5: Assign session predictions back to global array using the position mask
        all_spike_in_latent[sess_mask] = sess_predictions
        
        print(f"Session {sess} | Neurons: {X_sess.shape[1]} | Mean Test $R^2$: {np.mean(fold_r2_scores):.3f}")
        
    print("\nEncoding Complete! Output array matches original temporal layout.")
    return all_spike_in_latent
    

class FlightRQAAnalyzer:
    def __init__(self, min_len=20, max_len=100, threshold=0.05, l_min=2, v_min=2, dist_metric='euclidean'):
        """
        Args:
            min_len: Minimum number of time points for a valid flight.
            max_len: Maximum number of time points for a valid flight.
            threshold: Threshold (e.g., 0.05 = 5% density) to dynamically set epsilon.
            l_min: Minimum diagonal line length for Determinism (DET).
            v_min: Minimum vertical line length for Laminarity (LAM).
        """
        self.min_len = min_len
        self.max_len = max_len
        self.threshold = threshold
        self.l_min = l_min
        self.v_min = v_min
        self.dist_metric = dist_metric

    def _compute_single_flight_rqa(self, latents, theiler_window=2):
        """
        Computes the recurrence matrix and comprehensive RQA metrics for an isolated flight,
        properly adjusting for a specified Theiler window to remove short-term autocorrelation.
        
        Args:
            latents: array of shape (T, 32)
            theiler_window: int, number of diagonal steps to mask out (w=1 masks just the LOI)
        """
        T = latents.shape[0]
        
        # 1. Compute Pairwise Distance Matrix
        dist_matrix = squareform(pdist(latents, metric=self.dist_metric))
        
        # 2. Extract upper triangle distances strictly outside the Theiler window 
        # to ensure epsilon is dynamically fit only on non-trivial state pairs
        nontrivial_mask = np.triu(np.ones((T, T), dtype=bool), k=theiler_window)
        nontrivial_distances = dist_matrix[nontrivial_mask]
        
        if len(nontrivial_distances) == 0:
            return None
            
        # Target threshold is stored in self.threshold (e.g., 0.05)
        eps = np.percentile(nontrivial_distances, self.threshold * 100)
        
        # 3. Generate Binary Recurrence Plot (RP)
        RP_raw = (dist_matrix <= eps).astype(int)
        
        # Apply the Theiler Window mask to create the operational RP
        # This completely zeroes out the main diagonal and adjacent sub-diagonals
        full_theiler_mask = np.triu(np.ones((T, T)), k=theiler_window) + np.tril(np.ones((T, T)), k=-theiler_window)
        RP = RP_raw * full_theiler_mask
        
        # Total possible non-trivial points in the matrix after Theiler masking
        possible_points = np.sum(full_theiler_mask)
        total_recurrences = np.sum(RP)
        
        # Measure 1: Recurrence Rate (RR)
        rr = total_recurrences / possible_points if possible_points > 0 else 0.0
        
        if total_recurrences == 0:
            return {
                "RP": RP, "RR": rr, "DET": 0.0, "LAM": 0.0, 
                "L_avg": 0.0, "TT": 0.0, "DIV": 0.0, "ENTR": 0.0, "TREND": 0.0
            }

        # 4. Calculate Diagonal Lines Metrics (DET, L_avg, DIV, ENTR)
        diag_lengths = []
        for k in range(-T + 1, T):
            # Enforce Theiler window restriction across sub-diagonals
            if abs(k) < theiler_window: 
                continue
            diag = np.diag(RP, k=k)
            padded = np.convolve(diag, [1, -1])
            starts = np.where(padded == 1)[0]
            ends = np.where(padded == -1)[0]
            diag_lengths.extend(ends - starts)
            
        diag_lengths = np.array(diag_lengths)
        det_lines = diag_lengths[diag_lengths >= self.l_min]
        
        # Measure 2 & 4: DET and Averaged Diagonal Line Length (L_avg)
        det = np.sum(det_lines) / total_recurrences if len(det_lines) > 0 else 0.0
        l_avg = np.mean(det_lines) if len(det_lines) > 0 else 0.0
        
        # Measure 6: Divergence (DIV)
        l_max = np.max(diag_lengths) if len(diag_lengths) > 0 else 0
        div = 1.0 / l_max if l_max >= self.l_min else 0.0
        
        # Measure 7: Shannon Entropy of Diagonal Line Lengths (ENTR)
        if len(det_lines) > 0:
            # Count the frequency of unique line lengths
            _, counts = np.unique(det_lines, return_counts=True)
            probabilities = counts / len(det_lines)
            entr = -np.sum(probabilities * np.log(probabilities))
        else:
            entr = 0.0

        # 5. Calculate Vertical Lines Metrics (LAM, TT)
        vert_lengths = []
        for col in range(T):
            vertical = RP[:, col]
            padded = np.convolve(vertical, [1, -1])
            starts = np.where(padded == 1)[0]
            ends = np.where(padded == -1)[0]
            vert_lengths.extend(ends - starts)
            
        vert_lengths = np.array(vert_lengths)
        lam_lines = vert_lengths[vert_lengths >= self.v_min]
        
        # Measure 3 & 5: LAM and Trapping Time (TT)
        lam = np.sum(lam_lines) / total_recurrences if len(lam_lines) > 0 else 0.0
        tt = np.mean(lam_lines) if len(lam_lines) > 0 else 0.0

        # 6. Calculate TREND (Linear regression of local recurrence density vs. distance from LOI)
        # Trend measures nonstationarity / structural degradation over time distances
        # To prove that R_to_R has a fundamentally different dynamic independent of flight duration, 
        # you must normalize the time-separation axis ($\Delta t$) to relative behavior time 
        # (0% to 100% of flight length) before running the linear regression.
        distances = []
        local_densities = []
        
        # Iterate through sub-diagonals going outward from the Theiler limit
        for k in range(theiler_window, T - 1):
            # Sum upper and lower sub-diagonals at distance k
            diag_upper = np.diag(RP, k=k)
            diag_lower = np.diag(RP, k=-k)
            
            combined_recurrences = np.sum(diag_upper) + np.sum(diag_lower)
            combined_possible = len(diag_upper) + len(diag_lower)
            
            if combined_possible > 0:
                # Normalize the time distance by the absolute trajectory length T
                distances.append(k/T)
                local_densities.append(combined_recurrences / combined_possible)
                
        if len(distances) > 1:
            # Standard linear regression slope calculation
            slope, _ = np.polyfit(distances, local_densities, 1)
            # trend = slope * 1000  # Multiplied by 1000 per standard RQA scientific convention
            trend = slope  # No longer need to multiply by 1000 since scale is 0 to 1
        else:
            trend = 0.0

        return {
            "RP": RP,
            "RR": rr,
            "DET": det,
            "LAM": lam,
            "L_avg": l_avg,
            "TT": tt,
            "DIV": div,
            "ENTR": entr,
            "TREND": trend
        }

    def analyze(self, all_latents, all_rel_time, all_reward, all_perch, all_session):
        """
        Slices concatenated data and runs RQA across nested conditions:
        Environment (indoor vs outdoor) -> Task Type (R->R, R->P, P->R).
        """
        # Identify individual flight boundaries
        start_indices = np.where(all_rel_time == 0.0)[0]
        flight_boundaries = np.append(start_indices, len(all_rel_time))
        
        # Map conditions to human-readable strings
        task_mapping = {0: "R_to_R", 1: "R_to_P", 2: "P_to_R"}
        
        # Initialize structured nested dictionary output
        results = {
            "indoor": {name: [] for name in task_mapping.values()},
            "outdoor": {name: [] for name in task_mapping.values()}
        }
        
        for i in range(len(flight_boundaries) - 1):
            idx_start = flight_boundaries[i]
            idx_end = flight_boundaries[i+1]
            length = idx_end - idx_start
            
            # Filter by trajectory duration criteria
            if not (self.min_len <= length <= self.max_len):
                continue
                
            # Parse Environment Type
            session_name = all_session[idx_start]
            if "Dataset_1" in session_name or "Dataset_3" in session_name:
                env = "indoor"
            elif "Dataset_2" in session_name:
                env = "outdoor"
            else:
                continue # Unknown dataset group
                
            # Parse Flight Task Conditions
            is_start_reward = all_reward[idx_start]
            is_start_perch = all_perch[idx_start]
            is_end_reward = all_reward[idx_end - 1]
            is_end_perch = all_perch[idx_end - 1]
            
            if is_start_reward and is_end_reward:
                cond_id = 0
            elif is_start_reward and is_end_perch:
                cond_id = 1
            elif is_start_perch and is_end_reward:
                cond_id = 2
            else:
                continue # Exclude non-target trajectories (e.g., perch->perch)
                
            task_str = task_mapping[cond_id]
            flight_latents = all_latents[idx_start:idx_end, :]
            # If use all_true_spike (shape: 8332 by 266): 
            # Select all rows, but only columns (neurons) that are NOT completely NaN
            flight_latents = flight_latents[:, ~np.isnan(flight_latents).all(axis=0)]
            
            # Run RQA
            rqa_output = self._compute_single_flight_rqa(flight_latents)
            
            if rqa_output is not None:
                # Store the results metadata along with metrics
                results[env][task_str].append({
                    "flight_idx": i,
                    "global_indices": (idx_start, idx_end),
                    "trajectory_len": length,
                    "RP": rqa_output["RP"],
                    "DET": rqa_output["DET"],
                    "LAM": rqa_output["LAM"],
                    "L_avg": rqa_output["L_avg"],
                    "TT": rqa_output["TT"],
                    "DIV": rqa_output["DIV"],
                    "ENTR": rqa_output["ENTR"],
                    "TREND": rqa_output["TREND"],
                })
                
        return results
    
    def plot_rp(self, results, env='indoor', ncols = 5, seed = 10):
        task_keys = list(results[env].keys())
        nrows = len(task_keys)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
        for ii in range(nrows):
            np.random.seed(seed + ii)
            inds = np.random.choice(len(results[env][task_keys[ii]]), ncols, replace=False)
            for jj in range(ncols):
                # rqa_dataset['indoor']['reward_to_reward'][0]['RP']
                ax[ii, jj].matshow(results[env][task_keys[ii]][inds[jj]]['RP'], cmap='gray_r')
                ax[ii, jj].grid(True)
            ax[ii, 0].set_ylabel(f'{task_keys[ii]}')
        ax[0, 0].set_title(env)
        plt.tight_layout()
        
        return None
    
    def plot_rqa_distributions(self, results):
        """
        Generates a 2x3 subplot figure comparing the distributions of DET, LAM, and DIV
        with automated pairwise significance testing annotated on each plot.
        """
        from scipy.stats import mannwhitneyu
        from itertools import combinations

        # 1. Flatten the nested results dictionary into a tidy pandas DataFrame
        data_list = []
        for env, tasks in results.items():
            for task_str, flights in tasks.items():
                for flight in flights:
                    data_list.append({
                        "Environment": env.capitalize(),
                        "Flight Type": task_str.replace("_", " ").title(),
                        "DET": flight["DET"],
                        "LAM": flight["LAM"],
                        "L_avg": flight["L_avg"],
                        "TT": flight["TT"],
                        "DIV": flight["DIV"],
                        "ENTR": flight["ENTR"],
                        "TREND": flight["TREND"],
                    })
        
        df = pd.DataFrame(data_list)
        if df.empty:
            print("No data available to plot.")
            return

        # 2. Set up layout
        environments = ["Indoor", "Outdoor"]
        # We ignore L_avg, TT, DIV, ENTR which are influenced by the flight length.
        metrics = ["DET", "LAM", "TREND"]
        titles = ["DET", "LAM", "TREND"]
        palettes = {"Indoor": "Blues", "Outdoor": "Oranges"}
        fig, axes = plt.subplots(len(environments), len(metrics), figsize=(10, 6), sharex=True)
        
        # Helper function to convert p-values to your exact star mapping
        def get_star_string(p):
            if p < 0.0001: return '*****'
            elif p < 0.001: return '***'
            elif p < 0.01:  return '**'
            elif p < 0.05:  return '*'
            return None

        # 3. Loop through rows and columns
        for row_idx, env_name in enumerate(environments):
            df_env = df[df["Environment"] == env_name]
            
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                if df_env.empty:
                    continue

                # Ensure order of categories matches across plots
                flight_types = sorted(df_env["Flight Type"].unique())[::-1]
                
                sns.boxenplot(
                    data=df_env,
                    x="Flight Type",
                    y=metric,
                    order=flight_types,
                    hue="Flight Type",
                    legend=False,
                    palette=palettes[env_name],
                    ax=ax
                )
                
                # --- STATISTICAL ANNOTATION LOGIC ---
                # Find the maximum y-value currently in the data to place annotations dynamically
                y_max_base = df_env[metric].max()
                y_range = y_max_base - df_env[metric].min()
                if np.isnan(y_range) or y_range == 0: y_range = 1.0
                
                # Generate all unique pairs of categories
                pairs = list(combinations(flight_types, 2))
                annotation_count = 0
                
                for pair in pairs:
                    # Isolate data arrays for the two categories
                    group1 = df_env[df_env["Flight Type"] == pair[0]][metric].dropna().values
                    group2 = df_env[df_env["Flight Type"] == pair[1]][metric].dropna().values
                    
                    # We need at least 3 samples per group to run a meaningful comparison
                    if len(group1) < 3 or len(group2) < 3:
                        continue
                        
                    # Perform Mann-Whitney U test
                    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    star_str = get_star_string(p_val)
                    
                    if star_str: # Only plot if statistically significant
                        # Find indices to know where to draw lines horizontally
                        idx1 = flight_types.index(pair[0])
                        idx2 = flight_types.index(pair[1])
                        
                        # Dynamically step up the height of each bar to prevent overlapping lines
                        y_line = y_max_base + (0.08 * y_range) + (annotation_count * 0.12 * y_range)
                        h = 0.02 * y_range # Tick height
                        
                        # Draw the bracket line
                        ax.plot([idx1, idx1, idx2, idx2], [y_line-h, y_line, y_line, y_line-h], lw=1.5, c='k')
                        # Draw the star text centered over the line
                        ax.text((idx1 + idx2) * 0.5, y_line + (0.005 * y_range), star_str, ha='center', va='bottom', 
                            color='k', fontsize=12, fontweight='bold')
                        
                        annotation_count += 1
                
                # Adjust Y-limits to fit the annotations without clipping
                if annotation_count > 0:
                    ax.set_ylim(bottom=df_env[metric].min() - (0.05 * y_range), 
                                top=y_max_base + (annotation_count * 0.15 * y_range))
                
                # Aesthetics
                if row_idx == 0:
                    ax.set_title(titles[col_idx], fontsize=14, pad=15)
                
                # ax.set_ylabel(f"{metric} ({env_name})", fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(env_name)
                else:
                    ax.set_ylabel("")

                
                if row_idx == 1:
                    ax.set_xlabel("")
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.set_xlabel("")
                    
                ax.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return None
    

class FlightSharedStates():
    def __init__(self, all_latents, all_rel_time, all_reward, all_perch, all_session, 
                 MIN_LEN=20, MAX_LEN=100):
        """
        Parses concatenated data arrays, separates them into individual flight trials, 
        filters by length, and groups them by environment and functional type.
        """
        self.all_latents = all_latents
        self.all_rel_time = all_rel_time
        
        # Dictionary structure: env -> function_type -> list of flight dicts
        # Environments: 'indoor' (Dataset_1/3), 'outdoor' (Dataset_2)
        # Function types: 0 (R->R), 1 (R->P), 2 (P->R), 3 (P->P), 4 (Others)
        self.grouped_flights = {
            'indoor': {0: [], 1: [], 2: [], 3: [], 4: []},
            'outdoor': {0: [], 1: [], 2: [], 3: [], 4: []}
        }
        
        self._parse_and_group_flights(all_reward, all_perch, all_session, MIN_LEN, MAX_LEN)

    def _parse_and_group_flights(self, all_reward, all_perch, all_session, MIN_LEN, MAX_LEN):
        # Identify start boundaries where relative time resets to 0.0
        start_indices = np.where(self.all_rel_time == 0.0)[0]
        flight_boundaries = np.append(start_indices, len(self.all_rel_time))
        
        for i in range(len(flight_boundaries) - 1):
            idx_start = flight_boundaries[i]
            idx_end = flight_boundaries[i+1]
            
            # 1. Filter by Flight Trajectory Length
            length = idx_end - idx_start
            if not (MIN_LEN <= length <= MAX_LEN):
                continue
                
            # 2. Determine Environment Type
            session_name = all_session[idx_start]
            if "Dataset_1" in session_name or "Dataset_3" in session_name:
                env = 'indoor'
            elif "Dataset_2" in session_name:
                env = 'outdoor'
            else:
                continue # Skip unrecognized datasets
                
            # 3. Classify Flight Function Type
            is_start_reward = all_reward[idx_start]
            is_start_perch = all_perch[idx_start]
            is_end_reward = all_reward[idx_end - 1]
            is_end_perch = all_perch[idx_end - 1]
            
            if is_start_reward and is_end_reward:
                cond = 0  # R -> R
            elif is_start_reward and is_end_perch:
                cond = 1  # R -> P
            elif is_start_perch and is_end_reward:
                cond = 2  # P -> R
            elif is_start_perch and is_end_perch:
                cond = 3  # P -> P
            else:
                cond = 4 # Others
                
            # Pack flight data along with its global time indices
            flight_dict = {
                'latent': self.all_latents[idx_start:idx_end, :],
                'global_indices': np.arange(idx_start, idx_end)
            }
            
            self.grouped_flights[env][cond].append(flight_dict)
            
        # Print summary statistics upon initialization
        for env in ['indoor', 'outdoor']:
            print(f"--- {env.capitalize()} Environment ---")
            print(f"  Type 0 (R->R): {len(self.grouped_flights[env][0])} flights")
            print(f"  Type 1 (R->P): {len(self.grouped_flights[env][1])} flights")
            print(f"  Type 2 (P->R): {len(self.grouped_flights[env][2])} flights")
            print(f"  Type 3 (P->P): {len(self.grouped_flights[env][3])} flights")
            print(f"  Type 4 (Others): {len(self.grouped_flights[env][4])} flights")

    def find_shared_states(self, env='indoor', type_A=1, type_B=2, metric='cosine', thresh='min', num_pairs=500, seed=42):
        """
        Systematically random-samples pairs of flights from two function types, 
        computes their cross-distance, and extracts all global indices where the 
        pairwise distance falls below the specified percentile threshold.
        """
        from scipy.spatial.distance import cdist
        print(f'\nEnvironment: {env}: Type {type_A} vs. Type {type_B}')

        np.random.seed(seed)
        
        flights_A = self.grouped_flights[env][type_A]
        flights_B = self.grouped_flights[env][type_B]
        
        if len(flights_A) == 0 or len(flights_B) == 0:
            raise ValueError(f"One or both selected flight types contain 0 trajectories in the '{env}' environment.")
            
        pooled_global_indices = []
        all_thresholded_distances = []
        
        for _ in range(num_pairs):
            # Randomly select one flight from each type
            fA = np.random.choice(flights_A)
            fB = np.random.choice(flights_B)
            
            # Compute pairwise 32D distance matrix (Shape: Len_A x Len_B) using specified metric
            dist_matrix = cdist(fA['latent'], fB['latent'], metric=metric)

            if thresh == 'min':
                eps = np.min(dist_matrix)
            else:
                # Dynamically calculate the cutoff value for the bottom X% of distances for this pair
                eps = np.percentile(dist_matrix, thresh * 100)
            
            # Find all matrix coordinate pairs matching the threshold criteria
            idx_A_coords, idx_B_coords = np.where(dist_matrix <= eps)
            
            # Extract the actual distance values for tracking/diagnostics
            matching_distances = dist_matrix[idx_A_coords, idx_B_coords]
            all_thresholded_distances.extend(matching_distances)
            
            # Map the local flight time indices back to their absolute indices in all_latents
            global_idx_A = fA['global_indices'][idx_A_coords]
            global_idx_B = fB['global_indices'][idx_B_coords]
            
            pooled_global_indices.extend(global_idx_A)
            pooled_global_indices.extend(global_idx_B)
            
        # Deduplicate indices since overlapping pairs or adjacent time points can occur frequently
        unique_shared_indices = np.unique(pooled_global_indices)

        # Extract the latent vectors for just your discovered bottleneck points
        bottleneck_latents = self.all_latents[unique_shared_indices]
        # Compute the average distance between ALL discovered bottleneck points
        internal_distances = pdist(bottleneck_latents, metric=metric)
        np.random.seed(seed + 10)
        internal_distances = np.random.choice(internal_distances, num_pairs, replace=False)

        # NULL distances: randomly select pairs of points from flights_A and flights_B
        # Pre-pool all latent points from flights_A and flights_B to efficiently sample null pairs
        pooled_latents_A = np.vstack([f['latent'] for f in flights_A])
        pooled_latents_B = np.vstack([f['latent'] for f in flights_B])
        np.random.seed(seed + 100)
        idx_1 = np.random.choice(len(pooled_latents_A), num_pairs, replace=True)
        idx_2 = np.random.choice(len(pooled_latents_B), num_pairs, replace=True)
        # Compute baseline background distances
        null_distances = [cdist(pooled_latents_A[i:i+1], pooled_latents_B[j:j+1], metric=metric)[0,0] for i, j in zip(idx_1, idx_2)]
        
        results = {
            'shared_indices': unique_shared_indices,
            'pair_thresholded_distances': np.array(all_thresholded_distances),
            'mean_thresholded_distance': np.mean(all_thresholded_distances) if all_thresholded_distances else None,
            'total_points_collected': len(unique_shared_indices),
            'internal_distances': np.array(internal_distances),
            'pair_null_distances': np.array(null_distances),
        }
        
        return results
    

class SharedStatesSingleNeuron:
    def __init__(self, all_true_spike, all_session, bottleneck_indices, grouped_flights, env='indoor', flight_types=[1, 2]):
        """
        Parameters:
        -----------
        all_true_spike : np.ndarray
            Shape (Total_Time_Points, Max_Neurons), contains raw spike counts/rates with NaN padding.
        all_session : np.ndarray
            Shape (Total_Time_Points,), string array of session identifiers for each time point.
        bottleneck_indices : np.ndarray or list
            The global unique shared indices discovered from your low-dim PC analysis.
        grouped_flights : dict
            The nested dictionary (FlightSharedStates.grouped_flights) containing flight trajectories.
        env : str
            The specific environment to isolate (e.g., 'indoor').
        flight_types : list of int
            The functional types of flights to include [0 (R->R), 1 (R->P), 2 (P->R), 3 (P->P), 4 (Others)].
        """
        self.all_true_spike = all_true_spike
        self.all_session = all_session
        self.bottleneck_set = set(bottleneck_indices)
        
        # 1. Retain individual flight structures grouped by flight function types
        self.selected_flights = []
        selected_flight_indices = []
        
        for f_type in flight_types:
            flights = grouped_flights[env].get(f_type, [])
            for flight in flights:
                self.selected_flights.append(flight)
                selected_flight_indices.extend(flight['global_indices'])
                
        # Convert to a set for O(1) lookups during pooled baseline processing
        self.selected_flight_set = set(selected_flight_indices)
        
        # 2. Determine active sessions based on the global indices of chosen flights
        self.active_sessions = np.unique(all_session[list(self.selected_flight_set)])

    def analyze_session_neurons(self, session_name, p_threshold=0.05):
        """
        # Old:
        # Finds neurons within a session that pass the statistical modulation t-test 
        # AND strictly satisfy the trial-by-trial bottleneck participation rule.

        Iterate through each distinct flight belonging to the session and verifies 
        if the neuron spikes at least once (>=1) within the bottleneck zone on every 
        single flight to qualify as an anchor cell, and is completely silent (==0) within 
        the bottleneck zone on every single flight to qualify as a suppressed cell.
        """
        # Get all global indices for this session
        session_global_indices = np.where(self.all_session == session_name)[0]
        session_global_set = set(session_global_indices)
        
        # Filter down to the flights that belong strictly to THIS session
        session_flights = []
        for flight in self.selected_flights:
            # Check the first index to verify session membership
            if flight['global_indices'][0] in session_global_set:
                session_flights.append(flight)
                
        if len(session_flights) == 0:
            return pd.DataFrame()
            
        # Isolate pooled session indices inside vs outside the hub for the background t-test
        session_flight_indices = [idx for idx in session_global_indices if idx in self.selected_flight_set]
        inside_indices = [idx for idx in session_flight_indices if idx in self.bottleneck_set]
        outside_indices = [idx for idx in session_flight_indices if idx not in self.bottleneck_set]
        
        if len(inside_indices) < 3 or len(outside_indices) < 3:
            return pd.DataFrame()
            
        # Extract pooled session spikes
        spikes_inside = self.all_true_spike[inside_indices, :]
        spikes_outside = self.all_true_spike[outside_indices, :]
        
        # Filter out NaN columns (neurons not recorded in this session)
        valid_neuron_mask = ~np.all(np.isnan(spikes_inside), axis=0)
        valid_neuron_indices = np.where(valid_neuron_mask)[0]
        
        session_results = []
        
        for neuron_idx in valid_neuron_indices:
            rates_inside = spikes_inside[:, neuron_idx]
            rates_outside = spikes_outside[:, neuron_idx]
            
            rates_inside = rates_inside[~np.isnan(rates_inside)]
            rates_outside = rates_outside[~np.isnan(rates_outside)]
            
            if len(rates_inside) < 3 or len(rates_outside) < 3:
                continue
                
            # Perform baseline Welch's t-test for modulation direction
            t_stat, p_val = stats.ttest_ind(rates_inside, rates_outside, equal_var=False)
            mean_inside = np.mean(rates_inside)
            mean_outside = np.mean(rates_outside)
            
            # Initialize flight-by-flight tracking flags
            fired_in_all_bottlenecks = True
            silenced_in_all_bottlenecks = True
            valid_flight_count = 0
            
            # Evaluate the strict trial-by-trial rule over each individual flight
            for flight in session_flights:
                g_indices = flight['global_indices']
                # Isolate the bottleneck entries specific to this single flight
                flight_b_indices = [idx for idx in g_indices if idx in self.bottleneck_set]
                
                if len(flight_b_indices) == 0:
                    continue # Skip flights that do not enter the hub coordinates
                    
                valid_flight_count += 1
                flight_b_spikes = self.all_true_spike[flight_b_indices, neuron_idx]
                
                # Check metrics for this flight's bottleneck pass
                total_spikes_in_hub = np.nansum(flight_b_spikes)
                
                if total_spikes_in_hub < 1.0:
                    fired_in_all_bottlenecks = False # Broke the anchor rule
                if total_spikes_in_hub > 0.0:
                    silenced_in_all_bottlenecks = False # Broke the suppressed rule

            # If no flights actually intersected the hub, it cannot be classified
            if valid_flight_count == 0:
                fired_in_all_bottlenecks = False
                silenced_in_all_bottlenecks = False

            # # Enforce the dual constraints: statistical modulation AND 100% flight compliance
            # is_anchor = (p_val < p_threshold) and (mean_inside > mean_outside) and fired_in_all_bottlenecks
            # is_suppressed = (p_val < p_threshold) and (mean_inside < mean_outside) and silenced_in_all_bottlenecks
            # 100% flight compliance
            is_anchor = bool(fired_in_all_bottlenecks)
            is_suppressed = bool(silenced_in_all_bottlenecks)
            
            session_results.append({
                'neuron_idx_global': neuron_idx,
                'mean_inside_hub': mean_inside,
                'mean_outside_hub': mean_outside,
                'fold_change': mean_inside / (mean_outside + 1e-5),
                'p_value': p_val,
                'is_anchor': is_anchor, # is_anchor
                'is_suppressed': is_suppressed, # is_suppressed
            })
            
        return pd.DataFrame(session_results)

    def run_meta_analysis(self, p_threshold=0.05):
        """
        Loops through active sessions containing the filtered flights and pools results.
        """
        summary_data = []
        all_session_dfs = {}
        
        for session in self.active_sessions:
            df_session = self.analyze_session_neurons(session, p_threshold=p_threshold)
            
            if df_session.empty:
                continue
                
            all_session_dfs[session] = df_session
            
            total_valid_cells = len(df_session)
            total_anchors = df_session['is_anchor'].astype(int).sum()
            total_suppressed = df_session['is_suppressed'].astype(int).sum()
            
            summary_data.append({
                'session': session,
                'total_recorded_neurons': total_valid_cells,
                'anchor_cells_count': total_anchors,
                'suppressed_cells_count': total_suppressed,
                'percentage_anchors': (total_anchors / total_valid_cells) * 100
            })
            
        return pd.DataFrame(summary_data), all_session_dfs


def plot_bottleneck_single_neuron(session_name, all_session, detailed_session_data, grouped_flights, 
                                     all_true_spike, bottleneck_indices, env='indoor', 
                                     flight_types=[1, 2], num_tuning_curves=10):
    """
    Generates figures demonstrating single-neuron reliability at the bottleneck.
    
    1. A Sorted Raster Heatmap: Shows a sequential cascade of all valid neurons during a single flight,
       ordered by their peak firing time relative to the bottleneck.
    2. Cross-Flight Tuning Curves: Tracks individual anchor cells across multiple flights of 
       opposing types (Type 1 vs Type 2) to demonstrate directional invariance.
    """
    # ----------------------------------------------------
    # 0. Setup and Data Extraction
    # ----------------------------------------------------
    session_cells = detailed_session_data[session_name]
    anchor_cell_info = session_cells[session_cells['is_anchor']]
    anchor_global_indices = anchor_cell_info['neuron_idx_global'].values
    
    if len(anchor_global_indices) == 0:
        print(f"No anchor cells found in session {session_name} to visualize.")
        return

    # Gather all flights for this session from the chosen environment and types
    session_flights = []
    b_set = set(bottleneck_indices)
    
    for f_type in flight_types:
        for flight in grouped_flights[env].get(f_type, []):
            first_g_idx = flight['global_indices'][0]
            # CRITICAL FILTER: Ensure the flight belongs to THIS session
            if all_session[first_g_idx] == session_name:
                session_flights.append(flight)

    # ----------------------------------------------------
    # 1. Figure 1: The Sorted Raster Heatmap (Single Flight)
    # ----------------------------------------------------
    # Find up to 3 valid flights that pass through the bottleneck volume
    valid_flights = []
    for flight in session_flights:
        overlap = [idx for idx in flight['global_indices'] if idx in b_set]
        if len(overlap) >= 3:
            valid_flights.append(flight)
        if len(valid_flights) == 3:
            break
    if len(valid_flights) > 0:
        ncols = 3 if len(valid_flights) >= 3 else len(valid_flights)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5), sharey=True)
        if len(valid_flights) == 1:
            axes = [axes] # Ensure iterable array for consistency
            
        for idx, ax in enumerate(axes):
            g_indices = valid_flights[idx]['global_indices']
            flight_spikes = all_true_spike[g_indices, :]
            valid_cols = np.where(~np.all(np.isnan(flight_spikes), axis=0))[0]
            flight_spikes = flight_spikes[:, valid_cols]
            
            # Map valid columns to their global neuron identities for anchor matching
            global_cell_ids = valid_cols 
            
            in_bottleneck = np.array([idx_g in b_set for idx_g in g_indices])
            bottleneck_entry_idx = np.where(in_bottleneck)[0][0] if np.any(in_bottleneck) else len(g_indices)//2
            bottleneck_exit_idx = np.where(in_bottleneck)[0][-1] if np.any(in_bottleneck) else len(g_indices)//2
            
            norm_spikes = []
            peak_times = []
            for c in range(flight_spikes.shape[1]):
                # curve = pd.Series(flight_spikes[:, c]).rolling(window=3, min_periods=1, center=True).mean().values
                curve = flight_spikes[:, c]
                c_min, c_max = np.min(curve), np.max(curve)
                curve_norm = (curve - c_min) / (c_max - c_min + 1e-5)
                norm_spikes.append(curve_norm)
                peak_times.append(np.argmax(curve_norm))
                
            norm_spikes = np.array(norm_spikes)
            sort_order = np.argsort(peak_times)
            sorted_raster = norm_spikes[sort_order, :]
            
            # Find where the session's anchor cells ended up in this specific sorted layout
            sorted_global_ids = global_cell_ids[sort_order]
            is_anchor_sorted = np.isin(sorted_global_ids, anchor_global_indices)
            anchor_row_positions = np.where(is_anchor_sorted)[0]
            
            # Plot Heatmap
            sns.heatmap(sorted_raster, cmap='viridis', cbar=(idx == len(valid_flights)-1), 
                        cbar_kws={'label': 'Normalized responses'} if idx == len(valid_flights)-1 else {},
                        xticklabels=5, yticklabels=False, ax=ax)
            
            # Draw bottleneck trigger and align x-axis ticks to 0
            ax.axvline(x=bottleneck_entry_idx, color='white', linestyle='--', linewidth=2, label='Bottleneck')
            ax.axvline(x=bottleneck_exit_idx, color='white', linestyle='--', linewidth=2)
            # xticks = ax.get_xticks()
            # ax.set_xticks(xticks)
            # ax.set_xticklabels((xticks - bottleneck_entry_idx).astype(int))
            
            # Mark anchor cells on the y-axis with red horizontal tickers
            for row in anchor_row_positions:
                ax.text(-0.5, row + 0.5, '▶', color='red', ha='right', va='center', fontsize=8)
                
            ax.set_title(f"Flight #{idx+1}")
            ax.set_xlabel("Time Bins")
            if idx == 0:
                ax.set_ylabel("Sorted Neurons (▶ = Anchor)", labelpad=20)
                
        plt.suptitle(f"Population Cascades; Session: {session_name}", y=1.01)
        plt.tight_layout()

    # ----------------------------------------------------
    # 2. Figure 2: Cross-Flight Directional Invariance
    # ----------------------------------------------------
    # Let's take the top anchor cells with the largest fold change
    best_anchor_global_idxs = anchor_cell_info.sort_values(by='fold_change', ascending=False)['neuron_idx_global'].values
    
    # Separately plot profiles for Type 1 and Type 2 flights to verify alignment
    colors = {flight_types[0]: '#1f77b4', flight_types[1]: '#ff7f0e'}
    labels = {flight_types[0]: 'R -> P', flight_types[1]: 'P -> R'}
    # Create a set of starting indices for fast, error-free membership checking
    session_flight_starts = {f['global_indices'][0] for f in session_flights}

    ncols = 3
    fig, axes = plt.subplots(ncols=ncols, figsize=(4*ncols, 3))
    for ii in range(ncols):
        for f_type in flight_types:
            # type_flights = [f for f in grouped_flights[env].get(f_type, []) if f in session_flights]
            # Check matching starting index instead of comparing the raw dictionary objects
            type_flights = [f for f in grouped_flights[env].get(f_type, []) 
                            if f['global_indices'][0] in session_flight_starts]
            # Limit to requested number of curves for readability
            curves_plotted = 0
            
            for flight in type_flights:
                g_indices = flight['global_indices']
                in_b = np.array([idx in b_set for idx in g_indices])
                
                if not np.any(in_b):
                    continue
                    
                b_entry = np.where(in_b)[0][0]  # bottleneck (intersection hub, shared states) entry
                # Construct a relative time alignment window (-15 bins to +15 bins around entry)
                rel_time = np.arange(len(g_indices)) - b_entry
                neuron_activity = all_true_spike[g_indices, best_anchor_global_idxs[ii]]
                
                # Smooth curve
                # smoothed_activity = pd.Series(neuron_activity).rolling(window=3, min_periods=1, center=True).mean().values
                
                # Plot individual flight line centered at 0
                axes[ii].plot(rel_time, neuron_activity, color=colors[f_type], alpha=0.3, 
                        label=labels[f_type] if curves_plotted == 0 else "")
                curves_plotted += 1
                if curves_plotted >= num_tuning_curves:
                    break
        axes[ii].axvline(x=0, color='red', linestyle=':', linewidth=2)
        axes[ii].set_xlim(-10, 15)
        axes[ii].set_title(f"Cell #{best_anchor_global_idxs[ii]}")
        axes[ii].grid(True, alpha=0.2)
    axes[0].set_xlabel("Time Bins (Relative to Bottleneck Entry)")
    axes[0].set_ylabel("Spike Counts")
    axes[0].legend(loc='upper left')
    
    plt.suptitle(f"Anchor Cells Across Trajectories;Session: {session_name}", y=1.01)
    plt.tight_layout()

    return None


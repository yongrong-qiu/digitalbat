import numpy as np
from scipy.signal import butter, filtfilt, hilbert

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
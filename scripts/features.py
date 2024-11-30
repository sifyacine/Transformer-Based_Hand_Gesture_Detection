import numpy as np
from scipy.fft import fft

def extract_fft_features(windows):
    """
    Extract FFT features for each window.
    Parameters:
        windows (numpy.ndarray): Array of shape (num_windows, window_size, num_channels).
    Returns:
        numpy.ndarray: FFT features of shape (num_windows, window_size//2, num_channels).
    """
    print("Extracting FFT features...")
    fft_features = [np.abs(fft(window, axis=0))[:len(window) // 2] for window in windows]
    return np.array(fft_features)

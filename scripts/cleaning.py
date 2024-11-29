import numpy as np

def clean_data(emg_signals, labels):
    """
    Clean EMG signals by removing NaN values and low-quality trials.
    """
    print("Cleaning data...")
    # Remove rows with NaN values
    mask = ~np.isnan(emg_signals).any(axis=1)
    emg_signals = emg_signals[mask]
    labels = labels[mask]

    # Additional cleaning: Remove trials based on custom quality checks (if applicable)
    # Example: Implement checks based on signal-to-noise ratio (SNR) here.

    return emg_signals, labels

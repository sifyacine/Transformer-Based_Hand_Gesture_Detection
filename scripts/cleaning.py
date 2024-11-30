import numpy as np

def clean_data(emg_data, labels):
    """
    Cleans EMG data by handling NaN values, outliers, and unwanted signals.
    Args:
        emg_data (numpy.ndarray): Raw EMG data (samples, features).
        labels (numpy.ndarray): Corresponding labels (samples,).
    Returns:
        tuple: (cleaned_emg_data, cleaned_labels)
    """
    # Remove rows with NaN values
    mask = ~np.isnan(emg_data).any(axis=1)
    emg_data = emg_data[mask]
    labels = labels[mask]

    # Remove rows with extreme outliers (3 standard deviations)
    z_scores = np.abs((emg_data - np.mean(emg_data, axis=0)) / np.std(emg_data, axis=0))
    mask = np.all(z_scores < 3, axis=1)  # Keep rows where all features are within 3 stds
    emg_data = emg_data[mask]
    labels = labels[mask]

    return emg_data, labels

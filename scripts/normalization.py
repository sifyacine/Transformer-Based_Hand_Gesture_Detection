import numpy as np

def normalize_emg(emg_data):
    """
    Normalizes EMG data feature-wise to zero mean and unit variance.
    Args:
        emg_data (numpy.ndarray): Raw EMG data (samples, features).
    Returns:
        tuple: (normalized_emg_data, normalization_params)
            - normalized_emg_data: Normalized data.
            - normalization_params: Mean and std used for normalization.
    """
    mean = np.mean(emg_data, axis=0)
    std = np.std(emg_data, axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    normalized_emg_data = (emg_data - mean) / std
    return normalized_emg_data, {"mean": mean, "std": std}

import numpy as np

def create_sliding_windows(emg_data, labels, window_size=200, step_size=50):
    """
    Creates sliding windows from the EMG data and corresponding labels.
    Args:
        emg_data (numpy.ndarray): Normalized EMG data (samples, features).
        labels (numpy.ndarray): Corresponding labels (samples,).
        window_size (int): Size of each sliding window.
        step_size (int): Step size for sliding windows.
    Returns:
        tuple: (windows, window_labels)
            - windows: Sliding windows (num_windows, window_size, features).
            - window_labels: Labels for each window.
    """
    num_samples, num_features = emg_data.shape
    windows = []
    window_labels = []

    # Create sliding windows
    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        windows.append(emg_data[start_idx:end_idx])
        # Use the label at the center of the window as the window's label
        window_labels.append(labels[start_idx + window_size // 2])

    windows = np.array(windows)
    window_labels = np.array(window_labels)
    return windows, window_labels

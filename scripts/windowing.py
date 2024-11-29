import numpy as np

def create_sliding_windows(emg_signals, labels, window_size=200, overlap=100):
    """
    Create sliding windows with overlap from EMG signals.
    """
    print("Creating sliding windows...")
    step_size = window_size - overlap
    windows = []
    window_labels = []

    for i in range(0, len(emg_signals) - window_size, step_size):
        windows.append(emg_signals[i:i + window_size])
        # Use the middle label for the window
        window_labels.append(labels[i + window_size // 2])

    return np.array(windows), np.array(window_labels)

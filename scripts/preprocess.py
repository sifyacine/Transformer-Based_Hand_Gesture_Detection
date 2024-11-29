import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import joblib
from scipy.fft import fft  # For optional feature extraction
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'DB4')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 1. Data Cleaning
def clean_data(emg_signals, labels):
    """
    Clean data by removing NaN values and low-quality trials.
    """
    print("Cleaning data...")
    # Remove rows with NaN values
    mask = ~np.isnan(emg_signals).any(axis=1)
    emg_signals = emg_signals[mask]
    labels = labels[mask]

    # Additional cleaning: Remove trials with poor signal quality (example placeholder)
    # For instance, based on signal-to-noise ratio (SNR), implement threshold checks here.

    return emg_signals, labels

# 2. Data Normalization
def normalize_emg(emg_signals):
    """
    Normalize EMG signals to zero mean and unit variance per channel.
    """
    print("Normalizing data...")
    scaler = StandardScaler()  # Normalize each channel
    emg_signals = scaler.fit_transform(emg_signals)
    return emg_signals, scaler

# 3. Windowing the Data
def create_sliding_windows(emg_signals, labels, window_size=200, overlap=100):
    """
    Create sliding windows with overlap from the EMG signals.
    """
    print("Creating sliding windows...")
    step_size = window_size - overlap
    windows = []
    window_labels = []

    for i in range(0, len(emg_signals) - window_size, step_size):
        windows.append(emg_signals[i:i + window_size])
        # Assign label as the one at the center of the window
        window_labels.append(labels[i + window_size // 2])

    return np.array(windows), np.array(window_labels)

# 4. Feature Extraction (Optional)
def extract_fft_features(windows):
    """
    Extract FFT features for each window.
    """
    print("Extracting FFT features (optional)...")
    fft_features = [np.abs(fft(window, axis=0))[:len(window) // 2] for window in windows]
    return np.array(fft_features)

# 5. Positional Encoding
def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings for transformer input.
    """
    print("Generating positional encodings...")
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    encoding = np.zeros_like(angles)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Apply sine to even indices
    encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Apply cosine to odd indices
    return encoding

# 6. Preparing the Dataset
def prepare_dataset():
    """
    Full preprocessing pipeline for DB4 dataset.
    """
    # List all .mat files in the raw data directory
    mat_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.mat')]

    all_windows = []
    all_labels = []

    for mat_file in mat_files:
        file_path = os.path.join(RAW_DATA_DIR, mat_file)

        # Load EMG signals and labels
        print(f"Loading data from {file_path}...")
        data = loadmat(file_path)

        # Replace 'emg' and 'stimulus' with the correct keys for DB4
        emg_signals = data['emg']  # EMG signals
        labels = data['stimulus']  # Gesture labels

        # Flatten labels if needed
        if len(labels.shape) > 1:
            labels = labels.flatten()

        # Step 1: Clean data
        emg_signals, labels = clean_data(emg_signals, labels)

        # Step 2: Normalize data
        emg_signals, _ = normalize_emg(emg_signals)

        # Step 3: Create sliding windows
        windows, window_labels = create_sliding_windows(emg_signals, labels)

        # Step 4: (Optional) Extract FFT features
        # Uncomment if you want to include FFT features
        # windows = extract_fft_features(windows)

        # Collect all windows and labels
        all_windows.append(windows)
        all_labels.append(window_labels)

    # Combine all windows and labels
    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Step 5: Add positional encodings
    positional_encodings = positional_encoding(seq_len=all_windows.shape[1], d_model=all_windows.shape[2])
    all_windows += positional_encodings

    # Step 6: Split into train, validation, and test sets
    print("Splitting data into train, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(all_windows, all_labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save preprocessed data
    print("Saving preprocessed data...")
    joblib.dump((X_train, y_train), os.path.join(PROCESSED_DATA_DIR, 'train_data.pkl'))
    joblib.dump((X_val, y_val), os.path.join(PROCESSED_DATA_DIR, 'val_data.pkl'))
    joblib.dump((X_test, y_test), os.path.join(PROCESSED_DATA_DIR, 'test_data.pkl'))

    print("Data preparation complete. Processed data saved in 'data/processed/'.")

if __name__ == "__main__":
    prepare_dataset()

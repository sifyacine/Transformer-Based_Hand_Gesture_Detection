import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import joblib

from cleaning import clean_data
from normalization import normalize_emg
from windowing import create_sliding_windows
from features import extract_fft_features
from positional_encoding import positional_encoding

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'DB4')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def prepare_dataset():
    """
    Main pipeline to preprocess the DB4 dataset for transformer-based gesture detection.
    """
    mat_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.mat')]

    all_windows = []
    all_labels = []

    for mat_file in mat_files:
        file_path = os.path.join(RAW_DATA_DIR, mat_file)
        print(f"Loading data from {file_path}...")

        data = loadmat(file_path)
        emg_signals = data['emg']  # Adjust key as per DB4 structure
        labels = data['stimulus']  # Adjust key as per DB4 structure

        if len(labels.shape) > 1:
            labels = labels.flatten()

        emg_signals, labels = clean_data(emg_signals, labels)
        emg_signals, _ = normalize_emg(emg_signals)
        windows, window_labels = create_sliding_windows(emg_signals, labels)

        # Optional: Uncomment if FFT features are needed
        # windows = extract_fft_features(windows)

        all_windows.append(windows)
        all_labels.append(window_labels)

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Add positional encoding
    print("Adding positional encodings...")
    positional_encodings = positional_encoding(seq_len=all_windows.shape[1], d_model=all_windows.shape[2])
    all_windows += positional_encodings

    # Split data
    print("Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(all_windows, all_labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save processed data
    print("Saving processed data...")
    joblib.dump((X_train, y_train), os.path.join(PROCESSED_DATA_DIR, 'train_data.pkl'))
    joblib.dump((X_val, y_val), os.path.join(PROCESSED_DATA_DIR, 'val_data.pkl'))
    joblib.dump((X_test, y_test), os.path.join(PROCESSED_DATA_DIR, 'test_data.pkl'))

    print("Data preparation complete. Saved in 'data/processed/'.")

if __name__ == "__main__":
    prepare_dataset()

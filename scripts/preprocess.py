import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import joblib

from cleaning import clean_data
from normalization import normalize_emg
from windowing import create_sliding_windows
from positional_encoding import positional_encoding

# Define base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'DB4')  # Adjust DB name
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def prepare_dataset():
    """
    Main pipeline to preprocess the DB4 dataset for transformer-based gesture detection.
    This function supports hierarchical directories for subjects and files, with error handling.
    """
    all_windows = []
    all_labels = []

    # Traverse subjects and their .mat files
    for subject_dir in os.listdir(RAW_DATA_DIR):
        subject_path = os.path.join(RAW_DATA_DIR, subject_dir)

        # Ensure it's a directory
        if not os.path.isdir(subject_path):
            continue

        print(f"Processing subject: {subject_dir}")

        # Find all .mat files in the subject's directory
        mat_files = [f for f in os.listdir(subject_path) if f.endswith('.mat')]

        for mat_file in mat_files:
            file_path = os.path.join(subject_path, mat_file)
            try:
                print(f"Loading data from {file_path}...")

                # Load data
                data = loadmat(file_path)
                emg_signals = data['emg']  # Adjust key as per DB4 structure
                labels = data['stimulus']  # Adjust key as per DB4 structure

                # Flatten labels if needed
                if len(labels.shape) > 1:
                    labels = labels.flatten()

                # Apply preprocessing steps
                emg_signals, labels = clean_data(emg_signals, labels)
                emg_signals, _ = normalize_emg(emg_signals)
                windows, window_labels = create_sliding_windows(emg_signals, labels)

                # Collect processed data
                all_windows.append(windows)
                all_labels.append(window_labels)

            except KeyError as e:
                print(f"KeyError: {e} in file {mat_file}. Skipping this file.")
            except ValueError as e:
                print(f"ValueError: {e} in file {mat_file}. Skipping this file.")
            except Exception as e:
                print(f"Unexpected error in file {mat_file}: {e}. Skipping this file.")

    # Combine all data
    try:
        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    except ValueError as e:
        print(f"ValueError during concatenation: {e}. Check input shapes.")
        return

    # Add positional encodings
    try:
        print("Adding positional encodings...")
        positional_encodings = positional_encoding(seq_len=all_windows.shape[1], d_model=all_windows.shape[2])
        all_windows += positional_encodings
    except Exception as e:
        print(f"Error adding positional encodings: {e}.")
        return

    # Split data
    try:
        print("Splitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(all_windows, all_labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    except Exception as e:
        print(f"Error during train-test split: {e}.")
        return

    # Save processed data
    try:
        print("Saving processed data...")
        joblib.dump((X_train, y_train), os.path.join(PROCESSED_DATA_DIR, 'train_data.pkl'))
        joblib.dump((X_val, y_val), os.path.join(PROCESSED_DATA_DIR, 'val_data.pkl'))
        joblib.dump((X_test, y_test), os.path.join(PROCESSED_DATA_DIR, 'test_data.pkl'))
        print("Data preparation complete. Processed data saved in 'data/processed/'.")
    except Exception as e:
        print(f"Error saving processed data: {e}.")

if __name__ == "__main__":
    prepare_dataset()

import os
import numpy as np
from data_loader import load_fold_data
from cleaning import clean_data
from normalization import normalize_emg
from windowing import create_sliding_windows
from positional_encoding import positional_encoding
import joblib

def preprocess_fold(fold_dir, raw_data_dir, processed_dir):
    """
    Preprocess data for a single fold.
    Args:
        fold_dir (str): Path to the fold directory (e.g., 'data/folds/fold_1').
        raw_data_dir (str): Path to the raw data directory (e.g., 'data/raw/DB4').
        processed_dir (str): Path to save processed data (e.g., 'data/processed/fold_1').
    """
    print(f"Processing fold: {fold_dir}")
    os.makedirs(processed_dir, exist_ok=True)

    # Load train and validation subjects
    try:
        train_subjects = np.load(os.path.join(fold_dir, "train_subjects.npy"))
        val_subjects = np.load(os.path.join(fold_dir, "val_subjects.npy"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing subject files in {fold_dir}: {e}")

    def preprocess_subjects(subjects, split_name):
        print(f"Preprocessing {split_name} subjects for fold: {fold_dir}")
        all_windows = []
        all_labels = []

        for subject in subjects:
            subject_dir = os.path.join(raw_data_dir, subject)
            if not os.path.exists(subject_dir):
                print(f"Warning: Subject directory {subject_dir} does not exist. Skipping...")
                continue

            mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]
            if not mat_files:
                print(f"Warning: No .mat files found in {subject_dir}. Skipping...")
                continue

            for mat_file in mat_files:
                file_path = os.path.join(subject_dir, mat_file)
                try:
                    emg_data, labels = load_fold_data(file_path, fold_idx=1)

                    # Apply preprocessing steps
                    emg_data, labels = clean_data(emg_data, labels)
                    emg_data, _ = normalize_emg(emg_data)
                    windows, window_labels = create_sliding_windows(emg_data, labels)

                    all_windows.append(windows)
                    all_labels.append(window_labels)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

        if all_windows:
            # Combine windows and add positional encodings
            all_windows = np.concatenate(all_windows, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            positional_enc = positional_encoding(all_windows.shape[1], all_windows.shape[2])
            all_windows += positional_enc

            # Save preprocessed data
            save_path = os.path.join(processed_dir, f"{split_name}_data.pkl")
            joblib.dump((all_windows, all_labels), save_path)
            print(f"Saved {split_name} data to {save_path}")
        else:
            print(f"No data to save for {split_name} in fold {fold_dir}")

    preprocess_subjects(train_subjects, "train")
    preprocess_subjects(val_subjects, "val")

if __name__ == "__main__":
    base_folds_dir = "data/folds"
    raw_data_dir = "data/raw/DB4"
    processed_base_dir = "data/processed"

    if not os.path.exists(base_folds_dir):
        raise FileNotFoundError(f"Folds directory {base_folds_dir} not found.")
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory {raw_data_dir} not found.")

    fold_dirs = [f for f in os.listdir(base_folds_dir) if f.startswith("fold_")]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {base_folds_dir}.")

    for fold_dir in fold_dirs:
        fold_path = os.path.join(base_folds_dir, fold_dir)
        processed_path = os.path.join(processed_base_dir, fold_dir)
        try:
            preprocess_fold(fold_path, raw_data_dir, processed_path)
        except Exception as e:
            print(f"Error processing fold {fold_dir}: {e}")

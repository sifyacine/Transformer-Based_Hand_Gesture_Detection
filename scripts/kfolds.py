import os
import numpy as np
import scipy.io
import pickle
from sklearn.model_selection import KFold

def mat_to_pkl(mat_file_path, pkl_file_path):
    """
    Convert a .mat file to .pkl format.
    Args:
        mat_file_path (str): Path to the .mat file.
        pkl_file_path (str): Path to save the .pkl file.
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Save the data as a .pkl file
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(mat_data, f)
    print(f"Converted {mat_file_path} to {pkl_file_path}")

def create_folds(raw_data_dir, output_dir, n_splits=5, random_state=42):
    """
    Create k-fold partitions from the raw dataset and convert .mat files to .pkl.
    Args:
        raw_data_dir (str): Path to the raw dataset (directory containing .mat files).
        output_dir (str): Path to save fold partitions and converted .pkl files.
        n_splits (int): Number of folds.
        random_state (int): Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of all .mat files in the raw_data_dir
    mat_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.mat')]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over the folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(mat_files)):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save the train and validation .mat files for the current fold
        train_files = [mat_files[i] for i in train_idx]
        val_files = [mat_files[i] for i in val_idx]

        np.save(os.path.join(fold_dir, "train_files.npy"), train_files)
        np.save(os.path.join(fold_dir, "val_files.npy"), val_files)

        # Convert .mat files to .pkl files
        for mat_file in train_files + val_files:
            mat_file_path = os.path.join(raw_data_dir, mat_file)
            pkl_file_path = os.path.join(fold_dir, mat_file.replace('.mat', '.pkl'))
            mat_to_pkl(mat_file_path, pkl_file_path)

        print(f"Fold {fold_idx + 1}: Train={len(train_files)} | Val={len(val_files)}")

if __name__ == "__main__":
    raw_data_dir = "data/raw/DB4"  # Path to raw dataset containing .mat files
    output_dir = "data/folds"      # Path to save fold partitions and converted .pkl files
    create_folds(raw_data_dir, output_dir, n_splits=5)

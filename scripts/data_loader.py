import joblib
import os

def load_fold_data(processed_dir, fold_idx):
    """
    Load train and validation datasets for a specific fold.
    Args:
        processed_dir (str): Path to the directory containing processed data.
        fold_idx (int): The index of the current fold.
    Returns:
        dict: A dictionary with keys 'train' and 'val', each containing (X, y).
    """
    train_file = os.path.join(processed_dir, f"train_fold_{fold_idx}.pkl")
    val_file = os.path.join(processed_dir, f"val_fold_{fold_idx}.pkl")

    X_train, y_train = joblib.load(train_file)
    X_val, y_val = joblib.load(val_file)

    print(f"Loaded fold {fold_idx} data:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
    }

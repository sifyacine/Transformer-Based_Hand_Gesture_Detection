import joblib
import os

def load_processed_data(processed_dir):
    """
    Load train, validation, and test datasets from the processed directory.
    Args:
        processed_dir (str): Path to the directory containing processed data.
    Returns:
        dict: A dictionary with keys 'train', 'val', 'test', each containing (X, y).
    """
    train_file = os.path.join(processed_dir, "train_data.pkl")
    val_file = os.path.join(processed_dir, "val_data.pkl")
    test_file = os.path.join(processed_dir, "test_data.pkl")

    X_train, y_train = joblib.load(train_file)
    X_val, y_val = joblib.load(val_file)
    X_test, y_test = joblib.load(test_file)

    print(f"Loaded data:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
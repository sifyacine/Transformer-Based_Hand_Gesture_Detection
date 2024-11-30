import matplotlib.pyplot as plt
import numpy as np

def visualize_sample(X, y, sample_idx):
    """
    Visualize a single sample (time-series window) with its label.
    Args:
        X (numpy.ndarray): Input data of shape (num_samples, window_size, num_features).
        y (numpy.ndarray): Labels of shape (num_samples,).
        sample_idx (int): Index of the sample to visualize.
    """
    single_window = X[sample_idx]
    label = y[sample_idx]

    plt.figure(figsize=(10, 6))
    for feature_idx in range(single_window.shape[1]):
        plt.plot(single_window[:, feature_idx], label=f"Feature {feature_idx + 1}")

    plt.title(f"Sample {sample_idx} - Label: {label}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def visualize_label_distribution(y_train, y_val, y_test):
    """
    Plot the distribution of labels for train, validation, and test sets.
    Args:
        y_train (numpy.ndarray): Training labels.
        y_val (numpy.ndarray): Validation labels.
        y_test (numpy.ndarray): Test labels.
    """
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    val_labels, val_counts = np.unique(y_val, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)

    plt.figure(figsize=(12, 6))
    plt.bar(train_labels, train_counts, alpha=0.6, label="Train")
    plt.bar(val_labels, val_counts, alpha=0.6, label="Validation")
    plt.bar(test_labels, test_counts, alpha=0.6, label="Test")

    plt.title("Label Distribution")
    plt.xlabel("Gesture Classes")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()


def visualize_multiple_samples(X, num_windows_to_plot=5):
    """
    Visualize multiple sliding windows from the dataset.
    Args:
        X (numpy.ndarray): Input data of shape (num_samples, window_size, num_features).
        num_windows_to_plot (int): Number of windows to visualize.
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_windows_to_plot):
        plt.subplot(num_windows_to_plot, 1, i + 1)
        for feature_idx in range(X.shape[2]):  # Loop over features
            plt.plot(X[i, :, feature_idx], label=f"Feature {feature_idx + 1}" if i == 0 else "")
        plt.title(f"Sliding Window {i + 1}")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()
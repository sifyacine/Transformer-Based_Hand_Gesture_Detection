import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    plt.bar(train_labels, train_counts, alpha=0.6, label="Train", width=0.2, align='center')
    plt.bar(val_labels, val_counts, alpha=0.6, label="Validation", width=0.2, align='edge')
    plt.bar(test_labels, test_counts, alpha=0.6, label="Test", width=0.2, align='edge')

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


def visualize_data_before_after_preprocessing(X_raw, X_processed, y_raw, y_processed, sample_idx=0):
    """
    Visualize the raw and processed data for comparison.
    Args:
        X_raw (numpy.ndarray): Raw data of shape (num_samples, window_size, num_features).
        X_processed (numpy.ndarray): Processed data of shape (num_samples, window_size, num_features).
        y_raw (numpy.ndarray): Raw labels.
        y_processed (numpy.ndarray): Processed labels.
        sample_idx (int): Index of the sample to visualize.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Raw data
    axs[0].set_title(f"Raw Sample {sample_idx} - Label: {y_raw[sample_idx]}")
    for feature_idx in range(X_raw.shape[2]):  # Corrected: Loop over features (X_raw.shape[2])
        axs[0].plot(X_raw[sample_idx, :, feature_idx], label=f"Feature {feature_idx + 1}")
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Processed data
    axs[1].set_title(f"Processed Sample {sample_idx} - Label: {y_processed[sample_idx]}")
    for feature_idx in range(X_processed.shape[2]):  # Corrected: Loop over features (X_processed.shape[2])
        axs[1].plot(X_processed[sample_idx, :, feature_idx], label=f"Feature {feature_idx + 1}")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def visualize_feature_correlations(X):
    """
    Visualize the correlations between features (channels).
    Args:
        X (numpy.ndarray): Input data of shape (num_samples, window_size, num_features).
    """
    feature_corr = np.corrcoef(X.reshape(-1, X.shape[2]).T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_corr, annot=True, cmap="coolwarm", xticklabels=[f"Feature {i+1}" for i in range(X.shape[2])], yticklabels=[f"Feature {i+1}" for i in range(X.shape[2])])
    plt.title("Feature Correlation Heatmap")
    plt.show()


def visualize_time_series_vs_label(X, y):
    """
    Visualize the time-series data of each class across time.
    Args:
        X (numpy.ndarray): Input data of shape (num_samples, window_size, num_features).
        y (numpy.ndarray): Labels of shape (num_samples,).
    """
    unique_labels = np.unique(y)
    
    plt.figure(figsize=(10, 6))
    
    for label in unique_labels:
        label_samples = X[y == label]
        mean_signal = label_samples.mean(axis=0)
        plt.plot(mean_signal[:, 0], label=f"Class {label} (Mean Signal)")

    plt.title("Mean Time Series for Each Gesture Class")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


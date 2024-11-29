from sklearn.preprocessing import StandardScaler

def normalize_emg(emg_signals):
    """
    Normalize EMG signals to zero mean and unit variance per channel.
    """
    print("Normalizing data...")
    scaler = StandardScaler()
    emg_signals = scaler.fit_transform(emg_signals)
    return emg_signals, scaler

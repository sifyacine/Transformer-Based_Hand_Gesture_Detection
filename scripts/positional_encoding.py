import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings for transformer input.
    """
    print("Generating positional encodings...")
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    encoding = np.zeros_like(angles)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Sine for even indices
    encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Cosine for odd indices
    return encoding

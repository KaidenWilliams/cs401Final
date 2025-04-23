import librosa
import numpy as np

# --- Helper Functions (compute_melspec, mono_to_color, crop_or_pad) ---
# (These functions remain the same as in your provided script)
def compute_melspec(audio, sr, n_mels, fmin, fmax):
    """Compute a mel-spectrogram."""
    melspec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, hop_length=1024
    )
    melspec = librosa.power_to_db(melspec)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """Convert mono audio to color image format"""
    X = X.astype(np.float32)
    mean = mean if mean is not None else X.mean()
    std = std if std is not None else X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
    else:
        V = np.zeros_like(X)
    V = V.astype(np.uint8)
    return np.stack([V, V, V], axis=2)


def crop_or_pad(audio, length, sr):
    """Crop or pad an audio sample to a fixed length."""
    if len(audio) < length:
        audio = np.pad(audio, (0, length - len(audio)))
    elif len(audio) > length:
        audio = audio[:length]
    return audio

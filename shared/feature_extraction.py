"""Ekstraksi 26 fitur audio untuk pipeline XGBoost."""
import numpy as np
import librosa


def extract_features_from_segment(segment: np.ndarray, sr: int = 16000,
                                   n_mfcc: int = 20,
                                   n_fft: int = 512,
                                   hop_length: int = 256) -> np.ndarray:
    """
    Ekstrak 26 fitur statistik dari satu segmen audio.

    Fitur:
        1   Chroma STFT (mean)
        1   RMS Energy (mean)
        3   Spectral: Centroid, Bandwidth, Rolloff (mean masing-masing)
        1   Zero Crossing Rate (mean)
        20  MFCC koefisien 1–20 (mean masing-masing)

    Returns:
        np.ndarray shape (26,)
    """
    features = []

    features.append(float(np.mean(librosa.feature.chroma_stft(
        y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length))))
    features.append(float(np.mean(librosa.feature.rms(
        y=segment, hop_length=hop_length))))
    features.append(float(np.mean(librosa.feature.spectral_centroid(
        y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length))))
    features.append(float(np.mean(librosa.feature.spectral_bandwidth(
        y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length))))
    features.append(float(np.mean(librosa.feature.spectral_rolloff(
        y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85))))
    features.append(float(np.mean(librosa.feature.zero_crossing_rate(
        y=segment, hop_length=hop_length))))

    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    features.extend([float(np.mean(row)) for row in mfccs])

    return np.array(features, dtype=np.float32)


def extract_features_from_file(file_path: str, sr: int = 16000,
                                segment_len: int = 16000,
                                n_mfcc: int = 20,
                                n_fft: int = 512,
                                hop_length: int = 256) -> np.ndarray:
    """
    Load audio, segmentasi 1 detik, ekstrak 26 fitur per segmen,
    kembalikan rata-rata semua segmen (1 vektor per file).

    Returns:
        np.ndarray shape (26,)
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)

    n_segments = len(audio) // segment_len
    if n_segments == 0:
        audio = np.pad(audio, (0, segment_len - len(audio)))
        n_segments = 1

    feats = [
        extract_features_from_segment(
            audio[i * segment_len: (i + 1) * segment_len],
            sr, n_mfcc, n_fft, hop_length,
        )
        for i in range(n_segments)
    ]
    return np.mean(feats, axis=0)

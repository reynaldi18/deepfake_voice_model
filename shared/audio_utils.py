"""Audio loading utilities shared across notebooks."""
import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_waveform(path: str, sample_rate: int, max_samples: int) -> torch.Tensor:
    """
    Load mono audio, resample to sample_rate, pad/crop to max_samples.
    Falls back to librosa if soundfile fails.

    Returns:
        Tensor shape [max_samples]
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # [C, T]
    except Exception:
        import librosa
        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        waveform = torch.from_numpy(data.astype(np.float32))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    waveform = waveform.squeeze(0)  # [T]

    n = waveform.shape[0]
    if n < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - n))
    else:
        waveform = waveform[:max_samples]

    return waveform

"""Dataset classes for audio anti-spoofing."""
import csv
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from .audio_utils import load_waveform


class AudioDataset(Dataset):
    """Raw waveform dataset — digunakan oleh AASIST."""

    def __init__(self, tsv_path: str, max_samples: int, sample_rate: int):
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        with open(tsv_path, "r", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f, delimiter="\t"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        waveform = load_waveform(record["file_path"], self.sample_rate, self.max_samples)
        label = 0 if record["label"] == "bonafide" else 1
        return waveform, torch.tensor(label, dtype=torch.long)


class LFCCDataset(Dataset):
    """LFCC feature dataset — digunakan oleh MoLEx."""

    def __init__(self, tsv_path: str, max_samples: int, sample_rate: int,
                 n_lfcc: int, n_filter: int, n_fft: int,
                 win_length: int, hop_length: int):
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            n_filter=n_filter,
            speckwargs={"n_fft": n_fft, "win_length": win_length, "hop_length": hop_length},
        )
        with open(tsv_path, "r", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f, delimiter="\t"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        waveform = load_waveform(record["file_path"], self.sample_rate, self.max_samples)
        lfcc = self.lfcc_transform(waveform.unsqueeze(0)).squeeze(0)  # [n_lfcc, frames]
        # Cepstral Mean-Variance Normalization per utterance
        mean = lfcc.mean(dim=1, keepdim=True)
        std  = lfcc.std(dim=1, keepdim=True).clamp(min=1e-8)
        lfcc = (lfcc - mean) / std
        label = 0 if record["label"] == "bonafide" else 1
        return lfcc, torch.tensor(label, dtype=torch.long)

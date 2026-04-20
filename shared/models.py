"""Anti-spoofing model architectures (AASIST & MoLEx)."""
import torch
import torch.nn as nn


# ── AASIST helpers ────────────────────────────────────────────────────────────

class SincConv(nn.Module):
    """Learnable sinc-based filterbank (layer pertama RawNet)."""
    def __init__(self, out_channels: int = 70, kernel_size: int = 1024):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size, stride=16,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(torch.abs(self.conv(x.unsqueeze(1)))))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.skip(x))


class AASIST(nn.Module):
    """
    AASIST-lite: kompatibel MPS.
    Pipeline: SincConv → ResBlock encoder → AdaptiveAvgPool → Classifier
    """
    def __init__(self):
        super().__init__()
        self.sinc    = SincConv(out_channels=70, kernel_size=1024)
        self.encoder = nn.Sequential(
            ResBlock(70,  128, 3), ResBlock(128, 128, 3),
            ResBlock(128, 256, 3), ResBlock(256, 256, 3),
            ResBlock(256, 512, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.head(self.pool(self.encoder(self.sinc(x))))


# ── MoLEx helpers ─────────────────────────────────────────────────────────────

class MFM(nn.Module):
    """Max-Feature-Map activation — ciri khas LCNN."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MFM()

    def forward(self, x):
        return self.mfm(self.bn(self.conv(x)))


class AttentionPool(nn.Module):
    """Learnable attention pooling sepanjang dimensi temporal."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class MoLEx(nn.Module):
    """
    MoLEx: LCNN (Light CNN) + Attention Pooling.
    Input: LFCC features [B, n_lfcc, frames]
    """
    def __init__(self, n_lfcc: int = 60):
        super().__init__()
        self.frontend = nn.Sequential(
            LCNNBlock(1,  16, kernel=5, padding=2),
            nn.MaxPool2d(2, 2),
            LCNNBlock(16, 32),
            nn.MaxPool2d(2, 2),
            LCNNBlock(32, 64),
            nn.MaxPool2d(2, 2),
            LCNNBlock(64, 64),
        )
        self.feat_dim = 64 * (n_lfcc // 8)
        self.pool = AttentionPool(self.feat_dim)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # [B, 1, n_lfcc, frames]
        x = self.frontend(x)        # [B, 64, freq', frames']
        B, C, F, T = x.shape
        x = x.view(B, C * F, T).permute(0, 2, 1)   # [B, T, feat_dim]
        return self.head(self.pool(x))

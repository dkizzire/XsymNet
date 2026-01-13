import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import math
from typing import Optional


class simulated_data(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.X)

def moving_average(x, win=31):
    """
    Simple 1D moving average for background estimation.

    x: (B, L) tensor
    returns: (B, L) smoothed baseline
    """
    pad = win // 2
    x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")  # (B,1,L+2*pad)
    kernel = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win
    bg = F.conv1d(x_pad, kernel)  # (B,1,L)
    return bg.squeeze(1)    

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, max_len=4096):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Distance bias b(|i - j|) per head
        self.max_len = max_len
        self.dist_bias = nn.Embedding(max_len, num_heads)

        self.attn_dropout = nn.Dropout(attn_dropout)

        # For logging / regularization
        self.last_attn_weights = None  # (B, H, L, L)

    def forward(self, x):
        """
        x: (B, C, L)
        Returns: (B, C, L)
        """
        batch_size, channels, seq_len = x.shape

        # --- Q/K from normalized features: shape-based attention ---
        x_norm = x - x.mean(dim=2, keepdim=True)
        std = x_norm.std(dim=2, keepdim=True) + 1e-6
        x_norm = x_norm / std

        # Q,K from normalized, V from raw
        x_qk = x_norm.permute(0, 2, 1)  # (B, L, C)
        x_v  = x.permute(0, 2, 1)       # (B, L, C)

        Q = self.query(x_qk).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_qk).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_v ).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # shapes: (B, H, L, D)

        # --- scaled dot-product attention ---
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        # --- distance bias b(|i-j|) ---
        idx = torch.arange(seq_len, device=x.device)
        dist = (idx[None, :] - idx[:, None]).abs().clamp(max=self.max_len - 1)  # (L, L)
        # dist_bias(dist): (L, L, H) -> (1, H, L, L)
        db = self.dist_bias(dist).permute(2, 0, 1).unsqueeze(0)
        attn_scores = attn_scores + db

        # --- softmax + dropout ---
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.attn_dropout(attn_weights)
        self.last_attn_weights = attn_weights

        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.output_proj(attn_output)  # (B, L, C)

        return attn_output.permute(0, 2, 1)  # (B, C, L)

class CNN_MultiHeadAttention(nn.Module):
    lap_kernel: torch.Tensor
    def __init__(self, num_classes, channel_1, channel_2,
                 kernel_1, kernel_2, stride_1, stride_2,
                 num_heads, linear_1):
        super().__init__()
        self.num_classes = num_classes

        # ---- fixed Laplacian kernel for peakness: [1, -2, 1] ----
        lap = torch.tensor([[[1.0, -2.0, 1.0]]], dtype=torch.float32)  # (out=1, in=1, k=3)
        self.register_buffer("lap_kernel", lap)

        # First conv now takes 4 channels: raw, log, derivative, peakness
        in_ch = 4
        self.conv1 = nn.Conv1d(
            in_channels=in_ch,
            out_channels=channel_1,
            kernel_size=kernel_1,
            stride=stride_1,
            padding=kernel_1 // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(channel_1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=channel_1,
            out_channels=channel_2,
            kernel_size=kernel_2,
            stride=stride_2,
            padding=kernel_2 // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

        # Transformer-style block around attention
        self.attn_norm = nn.LayerNorm(channel_2)
        self.ffn = nn.Sequential(
            nn.Linear(channel_2, 2 * channel_2),
            nn.ReLU(),
            nn.Linear(2 * channel_2, channel_2),
        )
        self.ffn_norm = nn.LayerNorm(channel_2)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

        # for regularizers / inspection
        self.last_attn_weights = None

    def _build_multichannel_input(self, x):
        """
        x: (B, L) intensity pattern
        Returns: (B, 4, L): [raw, log1p(raw_clamped), derivative, peakness]
        """
        # raw
        x_raw = x.unsqueeze(1)  # (B, 1, L)

        # finite difference derivative along L
        dx = x[:, 1:] - x[:, :-1]
        dx = F.pad(dx, (1, 0))      # keep length L
        dx = dx.unsqueeze(1)        # (B, 1, L)

        # log channel (avoid log(0))
        x_pos = torch.clamp(x, min=0.0)
        x_log = torch.log1p(x_pos).unsqueeze(1)  # (B, 1, L)

        # peakness via discrete Laplacian: large (positive) at sharp peaks
        # lap_kernel is registered buffer: (1,1,3)
        lap = F.conv1d(x_raw, self.lap_kernel, padding=1)  # (B, 1, L)
        peakness = F.relu(-lap)  # only negative curvature (peaks)

        # normalize peakness per pattern so it's relative, not absolute
        max_p = peakness.amax(dim=2, keepdim=True) + 1e-6
        peakness = peakness / max_p  # (B, 1, L) in [0,1]

        # stack all channels
        x_multi = torch.cat([x_raw, x_log, dx, peakness], dim=1)  # (B, 4, L)
        return x_multi

    def forward(self, x):
        """
        x: (B, L)
        """
        x = self._build_multichannel_input(x)  # (B, 4, L)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)  # (B, C2, L')

        # --- attention block ---
        attn_x = self.attention(x)     # (B, C2, L')
        self.last_attn_weights = self.attention.last_attn_weights
        x = x + attn_x                 # residual

        # LayerNorm + FFN on (B, L', C2)
        x = x.permute(0, 2, 1)         # (B, L', C2)
        x = self.attn_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)         # back to (B, C2, L')

        # --- classifier head (same as before) ---
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    
class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2)             # (B, C)
        w = self.fc1(w)
        w = F.relu(w, inplace=True)
        w = self.fc2(w)
        w = torch.sigmoid(w)          # (B, C)
        w = w.unsqueeze(-1)           # (B, C, 1)
        return x * w                  # (B, C, L)
    

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, L)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + x)  # residual
        return out
    
class BraggDilatedResNet(nn.Module):
    def __init__(self, num_classes, channel_1, channel_2,
                 kernel_1, kernel_2, stride_1, stride_2,
                 num_heads, linear_1):
        """
        Drop-in alternative to CNN_MultiHeadAttention, but:
        - uses multi-scale dilated residual convs instead of attention
        - still takes x: (B, L) as input
        - peak-detection agnostic
        """
        super().__init__()
        self.num_classes = num_classes

        # ---- multichannel input: raw, log, derivative ----
        in_ch = 3

        # choose base_channels from hyperparams (we reuse channel_2)
        base_channels = channel_2

        # ---- stem: first conv + pool ----
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=base_channels,
                kernel_size=kernel_1,
                stride=stride_1,
                padding=kernel_1 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # ---- stack of dilated residual blocks ----
        # bigger dilations => model can see long-range peak distances
        dilations = [1, 2, 4, 8, 16]
        self.blocks = nn.ModuleList([
            ResidualBlock1D(
                channels=base_channels,
                kernel_size=kernel_2,
                dilation=d
            )
            for d in dilations
        ])

        # optional extra downsampling if you want to compress length
        self.tail_pool = nn.MaxPool1d(kernel_size=3, stride=stride_2, padding=1)

        # ---- global pooled classifier head ----
        # fully conv → adaptive avg over Q, so classifier sees global pattern
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(base_channels, linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

        # keep API compatibility with previous attention model
        self.last_attn_weights = None  # always None here

    def _build_multichannel_input(self, x):
        """
        x: (B, L) intensity pattern
        Returns: (B, 3, L): [raw, log1p(raw_clamped), derivative]
        """
        # raw
        x_raw = x.unsqueeze(1)  # (B, 1, L)

        # finite difference derivative along L
        dx = x[:, 1:] - x[:, :-1]
        dx = F.pad(dx, (1, 0))  # keep length
        dx = dx.unsqueeze(1)    # (B, 1, L)

        # log channel (avoid log(0))
        x_pos = torch.clamp(x, min=0.0)
        x_log = torch.log1p(x_pos).unsqueeze(1)  # (B, 1, L)

        x_multi = torch.cat([x_raw, x_log, dx], dim=1)  # (B, 3, L)
        return x_multi

    def forward(self, x):
        """
        x: (B, L)
        """
        # ---- build multichannel representation ----
        x = self._build_multichannel_input(x)  # (B, 3, L)

        # ---- stem ----
        x = self.stem(x)  # (B, C, L')

        # ---- dilated residual blocks ----
        for block in self.blocks:
            x = block(x)  # (B, C, L')

        x = self.tail_pool(x)  # (B, C, L'')

        # ---- global pooling over Q ----
        x = self.global_pool(x)        # (B, C, 1)
        x = x.squeeze(-1)              # (B, C)

        # ---- classifier ----
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class BraggAugment:
    def __init__(self,
                 p=0.7,
                 intensity_noise_std=0.02,   # fraction of max
                 offset_std=0.02,            # fraction of max
                 shift_bins=2,
                 q_scale_eps=0.01):
        self.p = p
        self.intensity_noise_std = intensity_noise_std
        self.offset_std = offset_std
        self.shift_bins = shift_bins
        self.q_scale_eps = q_scale_eps

    def __call__(self, x):
        # x: (L,) 1D tensor
        if torch.rand(1) > self.p:
            return x

        L = x.shape[-1]
        x_max = x.abs().max().clamp(min=1e-6)

        # 1) Add multiplicative-ish intensity noise
        noise = torch.randn_like(x) * self.intensity_noise_std * x_max
        x = x + noise

        # 2) Add baseline offset
        offset = (2 * torch.rand(1) - 1) * self.offset_std * x_max
        x = x + offset

        # 3) Small shift in bins (position jitter)
        if self.shift_bins > 0:
            shift = int(torch.randint(-self.shift_bins, self.shift_bins + 1, (1,)))
            x = torch.roll(x, shifts=shift, dims=0)

        # 4) SMALL Q-scale drift (~1–2%) as warping
        if self.q_scale_eps > 0:
            eps = (2 * torch.rand(1) - 1) * self.q_scale_eps  # e.g. ±1%
            # Build a grid for 1D warp using grid_sample
            # treat x as (1,1,L)
            x_ = x.view(1, 1, -1, 1)        # (N=1,C=1,H=L,W=1)
            L = x_.shape[2]
            grid = torch.linspace(-1, 1, L, device=x.device)
            scaled_grid = grid * (1 + eps)
            grid_2d = torch.stack([scaled_grid, torch.zeros_like(scaled_grid)], dim=-1)
            grid_2d = grid_2d.view(1, L, 1, 2)
            x_warp = F.grid_sample(x_, grid_2d, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            x = x_warp.view(-1)

        return x

class BraggResNetMHA(nn.Module):
    def __init__(self, num_classes, channel_1, channel_2,
                 kernel_1, kernel_2, stride_1, stride_2,
                 num_heads, linear_1):
        """
        Bragg-aware, scale/offset-robust 1D CNN + self-attention.

        - Input:  x (B, L) raw intensities
        - Output: logits (B, num_classes)

        Internally:
        - Per-pattern min-max normalization (shape-based features).
        - 5 channels from a single pattern:
            [raw_norm, log1p(raw_norm), dI/dQ, background, residual_norm]
        - Stage 1: local conv + local residual stack at channel_1
        - Stage 2: conv2 (for CAM) + multi-scale dilated ResNet at channel_2
        - Sinusoidal positional encoding in Q before Multi-Head attention
        - Global pooling (mean + max over Q) → classifier head
        - Squeeze–Excite in both stages for channel-wise Bragg emphasis.

        Exposed for regularizers / analysis:
        - self.last_attn_weights   : attention maps
        - self.last_feat           : penultimate features (B, linear_1)
        - self.last_dof_pred       : scalar aux DOF regression (B,)
        - self.last_embedding      : normalized embedding (B, proj_dim)
        """
        super().__init__()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Bragg-aware front-end: 5 channels
        # [raw_norm, log1p, dI/dQ, background, residual_norm]
        # ------------------------------------------------------------------
        self.in_ch = 5

        # Laplacian kernel for peakness (we keep it in case you want it later)
        lap = torch.tensor([[[1.0, -2.0, 1.0]]], dtype=torch.float32)  # (1, 1, 3)
        self.register_buffer("lap_kernel", lap)

        # ------------------------------------------------------------------
        # Stage 1: local conv + residual stack @ channel_1
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv1d(
            in_channels=self.in_ch,
            out_channels=channel_1,
            kernel_size=kernel_1,
            stride=stride_1,
            padding=kernel_1 // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channel_1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Local residual blocks: focus on peak shapes and near-neighbor structure
        self.local_blocks = nn.Sequential(
            ResidualBlock1D(channel_1, kernel_size=kernel_2, dilation=1),
            ResidualBlock1D(channel_1, kernel_size=kernel_2, dilation=2),
        )
        self.local_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se_local = SqueezeExcite1D(channel_1)

        # ------------------------------------------------------------------
        # Stage 2: conv2 + multi-scale dilated ResNet @ channel_2
        # ------------------------------------------------------------------
        base_channels = channel_2

        # conv2 is used in your GradCAM code → keep name & role
        self.conv2 = nn.Conv1d(
            in_channels=channel_1,
            out_channels=base_channels,
            kernel_size=kernel_2,
            stride=stride_2,
            padding=kernel_2 // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(base_channels)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Multi-scale dilated blocks: long-range peak spacing & superstructure
        dilations = [1, 2, 4, 8]
        self.dilated_blocks = nn.ModuleList([
            ResidualBlock1D(
                channels=base_channels,
                kernel_size=kernel_2,
                dilation=d,
            )
            for d in dilations
        ])
        self.tail_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.se_dilated = SqueezeExcite1D(base_channels)

        # ------------------------------------------------------------------
        # Multi-Head self-attention over Q
        # ------------------------------------------------------------------
        self.attention = MultiheadSelfAttention(
            embed_dim=base_channels,
            num_heads=num_heads,
        )
        self.attn_norm = nn.LayerNorm(base_channels)
        self.ffn = nn.Sequential(
            nn.Linear(base_channels, 2 * base_channels),
            nn.ReLU(),
            nn.Linear(2 * base_channels, base_channels),
        )
        self.ffn_norm = nn.LayerNorm(base_channels)

        # ------------------------------------------------------------------
        # Global pooling + classifier & auxiliary heads
        # ------------------------------------------------------------------
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

        # Auxiliary heads (do NOT affect forward() output shape):
        #  - DOF regression (per pattern)
        #  - embedding projection (for later contrastive / analysis)
        self.dof_head = nn.Linear(linear_1, 1)
        proj_dim = min(128, linear_1)
        self.proj_head = nn.Linear(linear_1, proj_dim)

        # Caches for inspection / regularization
        self.last_attn_weights = None
        self.last_feat = None
        self.last_dof_pred = None
        self.last_embedding = None

    # ----------------------------------------------------------------------
    # Positional encoding in Q (so attention "knows" distance in Q-space)
    # ----------------------------------------------------------------------
    def _positional_encoding(self, L, C, device):
        """
        Sinusoidal positional encoding, shape: (1, C, L)
        """
        position = torch.arange(L, device=device).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, C, 2, device=device).float() * (-math.log(10000.0) / C)
        )  # (C/2,)

        pe = torch.zeros(L, C, device=device)  # (L, C)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)   # (1, C, L)
        return pe

    # ----------------------------------------------------------------------
    # Small helper: moving average background
    # ----------------------------------------------------------------------
    @staticmethod
    def _moving_average_1d(x, win=31):
        """
        x: (B, L)
        returns: bg (B, L), simple moving average using conv1d
        with zero padding (deterministic under torch.use_deterministic_algorithms).
        """
        # x: (B, L) -> (B,1,L)
        x_in = x.unsqueeze(1)                        # (B,1,L)
    
        # simple average kernel
        kernel = x.new_ones(1, 1, win) / win        # (1,1,win)
    
        pad = win // 2
        # conv1d with padding = pad uses deterministic zero padding
        bg = F.conv1d(x_in, kernel, padding=pad)    # (B,1,L)
    
        return bg.squeeze(1)                        # (B,L)

    # ----------------------------------------------------------------------
    # Bragg-aware multi-channel front-end, scale/offset robust
    # ----------------------------------------------------------------------
    def _build_multichannel_input(self, x, eps=1e-6):
        """
        x: (B, L) raw intensities

        Per-pattern:
        - min-max normalize: x_norm in [0, 1] → removes global scale & offset
        - channels:
            raw_norm,
            log1p(raw_norm),
            derivative dI/dQ,
            smooth background,
            residual_norm = (x_norm - bg) / max|resid|

        Returns:
            x_multi: (B, 5, L)
        """
        # per-pattern min-max normalization → shape-based features
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x_range = (x_max - x_min).clamp(min=eps)
        x_norm = (x - x_min) / x_range            # (B, L) in [0, 1]

        # raw (normalized)
        x_raw = x_norm.unsqueeze(1)               # (B, 1, L)

        # derivative in Q
        dx = x_norm[:, 1:] - x_norm[:, :-1]
        dx = F.pad(dx, (1, 0))                    # keep length
        dx = dx.unsqueeze(1)                      # (B, 1, L)

        # log channel
        x_log = torch.log1p(x_norm).unsqueeze(1)  # (B, 1, L)

        # smooth background via moving average (now deterministic)
        bg = self._moving_average_1d(x_norm, win=31).unsqueeze(1)   # (B,1,L)

        # residual = normalized peak-ish component
        resid = x_raw - bg                                          # (B,1,L)
        r_max = resid.abs().amax(dim=2, keepdim=True) + eps
        resid_norm = resid / r_max                                  # (B,1,L)

        # stack channels
        x_multi = torch.cat(
            [x_raw, x_log, dx, bg, resid_norm],
            dim=1
        )  # (B, 5, L)

        return x_multi

    # ----------------------------------------------------------------------
    # Forward: x -> logits, with all extra stuff cached in attributes
    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x: (B, L)  intensity vector

        Returns:
            logits: (B, num_classes)

        Side effects:
            self.last_attn_weights : (B, num_heads, L3, L3)
            self.last_feat         : (B, linear_1)
            self.last_dof_pred     : (B,)
            self.last_embedding    : (B, proj_dim)
        """
        # -------- Bragg-aware front-end --------
        x = self._build_multichannel_input(x)          # (B, 5, L)

        # -------- Stage 1: local conv + residual stack --------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)                           # (B, C1, L1)

        x = self.local_blocks(x)                       # (B, C1, L1)
        x = self.se_local(x)
        x = self.local_pool(x)                         # (B, C1, L1')

        # -------- Stage 2: conv2 + dilated ResNet --------
        x = self.conv2(x)                              # (B, C2, L2)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)                           # (B, C2, L2')

        for block in self.dilated_blocks:
            x = block(x)                               # (B, C2, L2')

        x = self.se_dilated(x)
        x = self.tail_pool(x)                          # (B, C2, L3)

        # -------- Positional encoding + Multi-Head attention --------
        B, C, L3 = x.shape
        pe = self._positional_encoding(L3, C, x.device)  # (1, C, L3)
        x = x + pe

        attn_out = self.attention(x)                   # (B, C2, L3)
        self.last_attn_weights = self.attention.last_attn_weights
        self.last_attn_weights = self.last_attn_weights.detach()  # avoid backprop error with multi-gpu by detaching
        x = x + attn_out                               # residual

        # Transformer-style FFN over (B, L3, C2)
        x = x.permute(0, 2, 1)                         # (B, L3, C2)
        x = self.attn_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)                         # (B, C2, L3)

        # -------- Global pooling over Q --------
        gap = x.mean(dim=2)                            # (B, C2)
        gmp = x.max(dim=2).values                      # (B, C2)
        feat = torch.cat([gap, gmp], dim=1)            # (B, 2*C2)

        # -------- Classifier + aux heads --------
        feat = self.dropout(feat)
        feat = F.relu(self.fc1(feat))                  # (B, linear_1)

        # cache features for later physics/use
        self.last_feat = feat
        self.last_dof_pred = self.dof_head(feat).squeeze(-1)        # (B,)
        self.last_embedding = F.normalize(self.proj_head(feat), dim=-1)

        logits = self.fc2(feat)                        # (B, num_classes)
        return logits

#^ ------------------------------------------------------------------------------------------------------
# * XsymNet with Multi-Head Attention  :: Single Convolutional Layer
#^ ------------------------------------------------------------------------------------------------------   


class CNN_MultiHeadAttention_1Conv(nn.Module):
    def __init__(self, num_classes, channel_2, kernel_2, stride_2, num_heads, linear_1):
        super().__init__()
        self.num_classes = num_classes

        self.conv2 = nn.Conv1d(1, channel_2, kernel_size=kernel_2, stride=stride_2, padding=kernel_2 // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channel_2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.attention = MultiheadSelfAttention(embed_dim=channel_2, num_heads=num_heads)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.LazyLinear(linear_1)
        self.fc2 = nn.Linear(linear_1, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        attn_x = self.attention(x)
        x = x + attn_x

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


#^ ------------------------------------------------------------------------------------------------------
# * Class Activation Map for specified target class
#^ ------------------------------------------------------------------------------------------------------   

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # Use full backward hook

    def generate_cam(self, input_tensor, target_class: int):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # --- Safety checks ---
        assert self.activations is not None, "Activations are None. Did the forward hook run?"
        assert self.gradients is not None, "Gradients are None. Did the backward hook run?"

        print(f"Activations Shape: {self.activations.shape}")  # Debugging
        print(f"Gradients Shape: {self.gradients.shape}")  # Debugging
        
        pooled_gradients = torch.mean(self.gradients, dim=[2])  # Global average pooling
        print(f"Pooled Gradients Shape: {pooled_gradients.shape}")  # Debugging
        
        cam = torch.zeros_like(self.activations[0, 0])
        for i in range(self.activations.shape[1]):
            cam += pooled_gradients[0, i] * self.activations[0, i]
        
        cam = F.relu(cam)  # Apply ReLU
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def generate_average_cam(self, input_batch, target_class):
        """
        Generate average CAM over a batch of samples from the same class.

        Args:
            input_batch (Tensor): Tensor of shape (N, seq_len), all from the same class.
            target_class (int): The target class index for CAM computation.

        Returns:
            avg_cam (np.ndarray): Averaged CAM over all input samples.
        """
        self.model.eval()
        cams = []

        for i in range(input_batch.shape[0]):
            input_tensor = input_batch[i:i+1]  # Keep batch dim
            output = self.model(input_tensor)
            self.model.zero_grad()

            class_score = output[:, target_class]
            class_score.backward(retain_graph=True)
            
            # --- Safety checks ---
            assert self.activations is not None, "Activations are None. Did the forward hook run?"
            assert self.gradients is not None, "Gradients are None. Did the backward hook run?"

            pooled_gradients = torch.mean(self.gradients, dim=[2])
            cam = torch.zeros_like(self.activations[0, 0])
            for j in range(self.activations.shape[1]):
                cam += pooled_gradients[0, j] * self.activations[0, j]

            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cams.append(cam.cpu().numpy())

        avg_cam = np.mean(cams, axis=0)
        return avg_cam

    # def generate_average_cam(self, input_batch, target_class):
    #     """
    #     Generate average CAM over a batch of samples from the same class.
    #     """
    #     self.model.eval()
    #     cams = []

    #     for i in range(input_batch.shape[0]):
    #         input_tensor = input_batch[i:i+1]  # shape: (1, seq_len)
    #         self.model.zero_grad()
    #         output = self.model(input_tensor)

    #         # Get scalar class score by summing
    #         class_score = output[:, target_class].sum()
    #         class_score.backward(retain_graph=True)

    #         pooled_gradients = torch.mean(self.gradients, dim=[2])
    #         cam = torch.zeros_like(self.activations[0, 0])
    #         for j in range(self.activations.shape[1]):
    #             cam += pooled_gradients[0, j] * self.activations[0, j]

    #         cam = F.relu(cam)
    #         cam = cam - cam.min()
    #         cam = cam / (cam.max() + 1e-8)
    #         cams.append(cam.cpu().numpy())

    #     avg_cam = np.mean(cams, axis=0)
    #     return avg_cam

    # def generate_average_cam(self, input_batch, target_class):
    #     """
    #     Generate average CAM over a batch of samples from the same class,
    #     in one forward/backward pass (faster than looping).
    #     """
    #     self.model.eval()
    #     self.model.zero_grad()

    #     # Forward pass (entire batch at once)
    #     output = self.model(input_batch)  # shape: (batch_size, num_classes)

    #     # Get scalar class score (sum over batch to backpropagate through all samples)
    #     class_score = output[:, target_class].sum()
    #     class_score.backward(retain_graph=False)  # gradients for all batch items

    #     # pooled_gradients: average gradient for each channel across batch + sequence
    #     # shape: (num_channels,)
    #     pooled_gradients = torch.mean(self.gradients, dim=[0, 2])

    #     # Ensure pooled_gradients and activations are on the same device and dtype
    #     act_device = self.activations.device if hasattr(self, 'activations') and self.activations is not None else pooled_gradients.device
    #     pooled_gradients = pooled_gradients.to(device=act_device, dtype=self.activations.dtype if self.activations is not None else pooled_gradients.dtype)

    #     # Weight activations with pooled gradients
    #     # self.activations shape: (batch_size, num_channels, seq_len)
    #     weighted_activations = (pooled_gradients[None, :, None] * self.activations).sum(dim=1)

    #     # Apply ReLU, normalize each CAM
    #     cams = F.relu(weighted_activations)
    #     cams = cams - cams.min(dim=1, keepdim=True)[0]
    #     cams = cams / (cams.max(dim=1, keepdim=True)[0] + 1e-8)

    #     # Average CAMs across the batch
    #     avg_cam = cams.mean(dim=0).cpu().numpy()  # shape: (seq_len,)

    #     return avg_cam


def plot_cam(xrd_pattern, cam, x_axis=None, x_units='ToF (μs)', save_name="grad_cam_plot.png", title=None, overlay=True, dpi=600):

    # quick x_units for pasting:
    # 2θ (degrees), ToF (μs), d-spacing (Å)

    cam = cam.squeeze()
    xrd_pattern = xrd_pattern.squeeze()

    if cam.shape[0] != xrd_pattern.shape[0]:
        print(f"Resizing CAM from {cam.shape} to {xrd_pattern.shape}")
        from scipy.ndimage import zoom
        cam = zoom(cam, zoom=xrd_pattern.shape[0] / cam.shape[0])
        cam = np.asarray(cam)
    # Normalize CAM for consistent coloring
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Use custom x-axis
    if x_axis is None:
        x_axis = np.arange(len(xrd_pattern))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)  # Set DPI and figsize

    if overlay:
        # Use LineCollection for smooth color transitions
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap('jet')

        # Create line segments
        points = np.array([x_axis, xrd_pattern], dtype=float).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments.tolist(), cmap=cmap, norm=norm, linewidth=1)  # Set LINE WIDTH
        lc.set_array(cam)

        ax.add_collection(lc)
        ax.autoscale()

        # Add colorbar
        fig.colorbar(lc, ax=ax, label="Activation Intensity")

    else:
        ax.plot(x_axis, xrd_pattern, label="XRD Pattern", color="black")
        scatter = ax.scatter(x_axis, xrd_pattern, c=cam, cmap="jet", alpha=0.5)
        fig.colorbar(scatter, ax=ax, label="Activation Intensity")
    
    if title is not None:
        ax.set_title(title, fontsize=14) 
    
    ax.set_xlabel(f'{x_units}', fontsize=12)  
    ax.set_ylabel("Intensity A.U.", fontsize=12)
    ax.legend()
    
    plt.savefig(save_name, dpi=dpi, bbox_inches='tight')  # High-resolution save
    plt.show()


#^ ------------------------------------------------------------------------------------------------------
# * Averaged Class Activation Map for specified target class
#^ ------------------------------------------------------------------------------------------------------   




def plot_avg_cam_with_expt(xrd_expt, avg_cam, x_axis=None, x_units='2θ (degrees)', save_name="avg_cam_plot.png", legend_label= '', title=None):
    from matplotlib import rcParams
    avg_cam = avg_cam.squeeze()
    xrd_expt = xrd_expt.squeeze()
   
    rcParams.update({
        "text.usetex": False,  # Use True for LaTeX rendering if desired
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "figure.dpi": 600,
        "savefig.bbox": "tight"
    })

    # Resize CAM if necessary
    if avg_cam.shape[0] != xrd_expt.shape[0]:
        print(f"Resizing CAM from {avg_cam.shape} to {xrd_expt.shape}")
        from scipy.ndimage import zoom
        avg_cam = zoom(avg_cam, zoom=xrd_expt.shape[0] / avg_cam.shape[0])
        avg_cam = np.asarray(avg_cam)
    # Normalize CAM
    avg_cam = (avg_cam - np.min(avg_cam)) / (np.max(avg_cam) + 1e-8)

    # x-axis
    if x_axis is None:
        x_axis = np.arange(len(xrd_expt))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
    
    # Plot CAM as background
    ax.imshow(avg_cam[np.newaxis, :], extent=(x_axis[0], x_axis[-1], np.min(xrd_expt), np.max(xrd_expt)),
              cmap='jet', aspect='auto', alpha=0.8, origin='lower', vmin=avg_cam.min(), vmax=avg_cam.max())

    # Plot experimental XRD as white line
    ax.plot(x_axis, xrd_expt, color='black', linewidth=1.5, label=legend_label)

    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'{x_units}', fontsize=14)
    ax.set_ylabel("Intensity A.U.", fontsize=14)
    ax.legend()
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    plt.show()


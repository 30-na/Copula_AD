"""Lightweight deep-learning baselines (Autoencoder, LSTM forecaster).

Kept deliberately small/fast (few epochs, small hidden sizes) since these
run per-dataset across the full noise-category sweep on CPU. Both are
scored on the SAME window_size grid as our own method, with thresholds
set from the (1-alpha) percentile of the TRAINING segment's own error
distribution -- no test-label peeking, same fairness rule as our other
baselines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

ALPHA = 0.05


def _zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std if std > 0 else 1.0)


def _windows_of(values: np.ndarray, window_size: int) -> np.ndarray:
    n = len(values) // window_size
    return np.stack([values[i * window_size : (i + 1) * window_size] for i in range(n)])


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        hidden = latent_dim * 2
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def autoencoder_scores(
    train_values: np.ndarray, test_values: np.ndarray, window_size: int, epochs: int = 15, seed: int = 0
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    mean, std = float(np.mean(train_values)), float(np.std(train_values))

    train_w = _zscore(_windows_of(train_values, window_size), mean, std)
    test_w = _zscore(_windows_of(test_values, window_size), mean, std)

    model = DenseAutoencoder(window_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(train_w, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(X) - X) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_err = ((model(X) - X) ** 2).mean(dim=1).numpy()
        Xt = torch.tensor(test_w, dtype=torch.float32)
        test_err = ((model(Xt) - Xt) ** 2).mean(dim=1).numpy()
    threshold = float(np.percentile(train_err, 100 * (1 - ALPHA)))
    return test_err, threshold


class ForecastLSTM(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


def lstm_scores(
    train_values: np.ndarray, test_values: np.ndarray, window_size: int,
    seq_len: int = 50, epochs: int = 8, seed: int = 0,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    mean, std = float(np.mean(train_values)), float(np.std(train_values))
    train_n = _zscore(train_values, mean, std)
    test_n = _zscore(test_values, mean, std)

    n_chunks = max(1, (len(train_n) - 1) // seq_len)
    chunks_in, chunks_tgt = [], []
    for i in range(n_chunks):
        seg = train_n[i * seq_len : (i + 1) * seq_len + 1]
        if len(seg) < 2:
            continue
        chunks_in.append(seg[:-1])
        chunks_tgt.append(seg[1:])

    model = ForecastLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    X = torch.tensor(np.stack(chunks_in), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.stack(chunks_tgt), dtype=torch.float32).unsqueeze(-1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()

    def per_sample_sq_err(values_n: np.ndarray) -> np.ndarray:
        seq = torch.tensor(values_n[:-1], dtype=torch.float32).view(1, -1, 1)
        tgt = torch.tensor(values_n[1:], dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            err = ((model(seq) - tgt) ** 2).squeeze(0).squeeze(-1).numpy()
        return np.concatenate([[err[0]], err])  # pad the first (unpredicted) sample

    def window_agg(err: np.ndarray) -> np.ndarray:
        n = len(err) // window_size
        return np.array([err[i * window_size : (i + 1) * window_size].mean() for i in range(n)])

    train_w_err = window_agg(per_sample_sq_err(train_n))
    test_w_err = window_agg(per_sample_sq_err(test_n))
    threshold = float(np.percentile(train_w_err, 100 * (1 - ALPHA)))
    return test_w_err, threshold

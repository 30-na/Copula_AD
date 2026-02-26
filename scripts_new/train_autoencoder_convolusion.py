import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path

class EEGWindowDatasetConv(Dataset):
    def __init__(self, folder, downsample=100):
        self.files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")
        ])
        self.downsample = downsample

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])[::self.downsample].astype(np.float32)  # (6000, 4)
        x = x.T  # → shape (4, 6000) for Conv1D
        return torch.from_numpy(x)

class EEGAutoencoderConv1D(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),  # -> (16, 3000)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),              # -> (32, 1500)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),              # -> (64, 750)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),  # -> (32, 1500)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),  # -> (16, 3000)
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1)  # -> (4, 6000)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load dataset
dataset = EEGWindowDatasetConv("data/mice/training_windows", downsample=100)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Setup
model = EEGAutoencoderConv1D()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
for epoch in range(500):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)  # shape: (B, 4, 6000)
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.6f}")

# Save model
torch.save(model.state_dict(), "models/eeg_autoencoder_conv1d.pth")

def evaluate_conv1d_model_on_file(
    dat_path,
    model,
    window_len=600000,
    step=20000,
    downsample=100,
    sampling_rate=2000,
    start_time="17:30:04"
):
    dat_path = Path(dat_path)
    model.eval()

    date_str = dat_path.name.split("TS_")[1][:10]
    recording_start = pd.to_datetime(f"{date_str} {start_time}")

    signal = np.memmap(dat_path, dtype=np.int16, mode='r').reshape(-1, 4)
    total_samples = signal.shape[0]

    losses = []
    timestamps = []

    for start in range(0, total_samples - window_len + 1, step):
        end = start + window_len
        window = signal[start:end]
        window_ds = window[::downsample].astype(np.float32).T  # shape (4, 6000)

        x = torch.from_numpy(window_ds).unsqueeze(0).to(next(model.parameters()).device)  # (1, 4, 6000)

        with torch.no_grad():
            recon = model(x)
            loss = torch.nn.functional.mse_loss(recon, x).item()

        time_offset_sec = start / sampling_rate
        timestamp = recording_start + pd.to_timedelta(time_offset_sec, unit='s')

        losses.append(loss)
        timestamps.append(timestamp)

    return pd.DataFrame({"timestamp": timestamps, "loss": losses})


result_0328 = evaluate_conv1d_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-03-28_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)

result_0330 = evaluate_conv1d_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-03-30_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)

result_0404 = evaluate_conv1d_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-04-04_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)

result_0325 = evaluate_conv1d_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-03-25_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)


plot_and_save_result_with_seizures(
    result_df=result_0328,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-03-28_17_30_04.txt",
    save_path="figures/reconstruction_loss_0328_conv1d.png"
)

plot_and_save_result_with_seizures(
    result_df=result_0330,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-03-30_17_30_04.txt",
    save_path="figures/reconstruction_loss_0330_conv1d.png"
)

plot_and_save_result_with_seizures(
    result_df=result_0404,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-04-04_17_30_04.txt",
    save_path="figures/reconstruction_loss_0404_conv1d.png"
)

plot_and_save_result_with_seizures(
    result_df=result_0325,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-03-25_17_30_04.txt",
    save_path="figures/reconstruction_loss_0325_conv1d.png"
)
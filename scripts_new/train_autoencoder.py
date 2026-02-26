import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import timedelta


def extract_windows_from_file(
    dat_filename,
    input_dir,
    output_dir,
    window_len=600000,
    overlap=20000,
    seizure_buffer=7200000
):
    dat_path = Path(input_dir) / dat_filename
    if not dat_path.exists():
        print(f"Missing .dat file: {dat_path}")
        return

    txt_filename = dat_filename.replace("_allCh.dat", ".txt").replace("_DOB_", "_DOB ")
    txt_path = Path(input_dir) / txt_filename
    if not txt_path.exists():
        print(f"Missing .txt file: {txt_path}")
        return

    df = pd.read_csv(txt_path, sep='\t', skiprows=6)
    seizure_starts = pd.to_datetime(
    df[df['Annotation'].str.lower().str.strip() == 'seizure starts']['Start Time'],
    format="%m/%d/%y %H:%M:%S.%f",
    errors="coerce"
)
    seizure_ends = pd.to_datetime(
        df[df['Annotation'].str.lower().str.strip() == 'seizure ends']['Start Time'],
        format="%m/%d/%y %H:%M:%S.%f",
        errors="coerce"
)


    date_str = dat_filename.split("TS_")[1][:10]
    recording_start = pd.to_datetime(f"{date_str} 17:30:04")

    signal = np.memmap(dat_path, dtype=np.int16, mode='r').reshape(-1, 4)
    total_samples = signal.shape[0]

    seizure_sample_ranges = []
    for s_time, e_time in zip(seizure_starts, seizure_ends):
        s_offset = int((s_time - recording_start).total_seconds() * 2000)
        e_offset = int((e_time - recording_start).total_seconds() * 2000)
        seizure_sample_ranges.append((
            max(0, s_offset - seizure_buffer),
            min(total_samples, e_offset + seizure_buffer)
        ))

    os.makedirs(output_dir, exist_ok=True)
    step = window_len - overlap
    i = 0

    for start in range(0, total_samples - window_len + 1, step):
        end = start + window_len
        if any(not (end < s or start > e) for s, e in seizure_sample_ranges):
            continue

        window = signal[start:end]
        out_path = os.path.join(output_dir, f"{dat_filename[:-4]}_window_{i:05d}.npy")
        np.save(out_path, window)
        i += 1

    print(f"{dat_filename}: saved {i} windows.")




def process_all_files(
    input_dir,
    output_dir,
    window_len=600000,
    overlap=20000,
    seizure_buffer=7200000
):
    dat_files = [
        f for f in os.listdir(input_dir)
        if f.endswith("_allCh.dat")
    ]

    for dat_file in tqdm(sorted(dat_files), desc="Processing .dat files"):
        extract_windows_from_file(
            dat_filename=dat_file,
            input_dir=input_dir,
            output_dir=output_dir,
            window_len=window_len,
            overlap=overlap,
            seizure_buffer=seizure_buffer
        )

process_all_files("data/mice/train", "data/mice/training_windows")



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class EEGWindowDataset(Dataset):
    def __init__(self, folder, downsample=1000):
        self.files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")
        ])
        self.downsample = downsample

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])[::self.downsample]  # downsample in time
        x = x.astype(np.float32).reshape(-1)
        return torch.from_numpy(x)


class EEGAutoencoder(nn.Module):
    def __init__(self, input_dim=24000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Load data
dataset = EEGWindowDataset("data/mice/training_windows", downsample=100)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Model setup
model = EEGAutoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
for epoch in range(300):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.6f}")

torch.save(model, "models/eeg_autoencoder_downsample1000.pth")

model = torch.load("models/eeg_autoencoder_downsample1000.pth")
model.eval()

def evaluate_model_on_file(
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

    # Parse datetime from filename
    date_str = dat_path.name.split("TS_")[1][:10]
    recording_start = pd.to_datetime(f"{date_str} {start_time}")

    # Load signal
    signal = np.memmap(dat_path, dtype=np.int16, mode='r').reshape(-1, 4)
    total_samples = signal.shape[0]

    losses = []
    timestamps = []

    for start in range(0, total_samples - window_len + 1, step):
        end = start + window_len
        window = signal[start:end]
        window_ds = window[::downsample].astype(np.float32).reshape(-1)

        x = torch.from_numpy(window_ds).unsqueeze(0)
        x = x.to(next(model.parameters()).device)

        with torch.no_grad():
            recon = model(x)
            loss = torch.nn.functional.mse_loss(recon, x).item()

        time_offset_sec = start / sampling_rate
        timestamp = recording_start + pd.to_timedelta(time_offset_sec, unit='s')

        losses.append(loss)
        timestamps.append(timestamp)

    return pd.DataFrame({"timestamp": timestamps, "loss": losses})


def plot_and_save_result_with_seizures(result_df, txt_path, save_path):
    # Load seizure annotations
    ann = pd.read_csv(txt_path, sep='\t', skiprows=6)
    seizure_starts = pd.to_datetime(
        ann[ann["Annotation"].str.lower().str.strip() == "seizure starts"]["Start Time"],
        format="%m/%d/%y %H:%M:%S.%f",
        errors="coerce"
    )
    seizure_ends = pd.to_datetime(
        ann[ann["Annotation"].str.lower().str.strip() == "seizure ends"]["Start Time"],
        format="%m/%d/%y %H:%M:%S.%f",
        errors="coerce"
    )

    # Plot
    plt.figure(figsize=(15, 5))
    plt.scatter(result_df["timestamp"], result_df["loss"], s=2, label="Reconstruction Loss")

    for s, e in zip(seizure_starts, seizure_ends):
        if pd.isnull(s) or pd.isnull(e): continue
        pre = s - timedelta(hours=1)
        plt.axvspan(pre, s, color="orange", alpha=0.3, label="1h Before Seizure")
        plt.axvspan(s, e, color="red", alpha=0.3, label="Seizure")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Reconstruction Loss with Seizure Periods")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path)
    plt.close()





result_0328 = evaluate_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-03-28_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)

result_0330 = evaluate_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-03-30_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)

result_0404 = evaluate_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-04-04_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)


result_0325 = evaluate_model_on_file(
    dat_path="data/mice/test/AC75a-5_DOB_072519_TS_2020-03-25_17_30_04_allCh.dat",
    model=model,
    window_len=600000,
    step=20000,
    downsample=100
)


plot_and_save_result_with_seizures(
    result_df=result_0328,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-03-28_17_30_04.txt",
    save_path="figures/reconstruction_loss_0328.png"
)

plot_and_save_result_with_seizures(
    result_df=result_0330,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-03-30_17_30_04.txt",
    save_path="figures/reconstruction_loss_0330.png"
)

plot_and_save_result_with_seizures(
    result_df=result_0404,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-04-04_17_30_04.txt",
    save_path="figures/reconstruction_loss_0404.png"
)

plot_and_save_result_with_seizures(
    result_df=result_0325,
    txt_path="data/mice/test/AC75a-5_DOB 072519_TS_2020-03-25_17_30_04.txt",
    save_path="figures/reconstruction_loss_0325.png"
)



import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class EarthquakeData:
    def __init__(self, base_path="data/earthquake"):
        self.base_path = base_path
        self.station_data = {}

    def read_file(self, file_path):
        df = pd.read_csv(
            file_path,
            skiprows=18,
            parse_dates=["Time"],
            low_memory=False
        )
        df.rename(columns={" Sample": "Sample"}, inplace=True)
        df['Sample'] = pd.to_numeric(df['Sample'], errors='coerce')
        df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
        df = df.dropna(subset=["Time"])
        df = df.set_index("Time")
        return df

    def break_segments(self, df):
        segments = []
        start = 0
        for i in range(1, len(df)):
            if df.index[i-1] >= df.index[i]:
                segments.append(df.iloc[start:i])
                start = i
        segments.append(df.iloc[start:])
        return segments



    def plot_segments_per_file(self, station_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        folder = os.path.join(self.base_path, station_name)
        files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])

        for file_idx, file in enumerate(files):
            full_path = os.path.join(folder, file)
            df = self.read_file(full_path)
            segments = self.break_segments(df)

            fig, axes = plt.subplots(len(segments), 1, figsize=(15, 1.5 * len(segments)), sharex=True)

            if len(segments) == 1:
                axes = [axes]

            for i, seg in enumerate(segments):
                axes[i].scatter(seg.index, seg['Sample'], s=0.2, color='black')
                axes[i].set_ylabel(f"Seg {i}", rotation=0, labelpad=20)
                axes[i].tick_params(axis='y', labelsize=6)

            axes[-1].set_xlabel("Time")
            plt.suptitle(f"{station_name.upper()} - {file}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{station_name}_file{file_idx}_segments.png"), dpi=300)
            plt.close()


    def load_station(self, station_name):
        folder = os.path.join(self.base_path, station_name)
        files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])

        segments_dict = {}
        for file_idx, file in enumerate(tqdm(files, desc=f"Reading {station_name}")):
            full_path = os.path.join(folder, file)
            df = self.read_file(full_path)
            segments = self.break_segments(df)
            print(len(segments))

            for seg_idx, segment in enumerate(segments):
                label = f"file{file_idx}_seg{seg_idx}"
                segments_dict[label] = segment["Sample"]

        # Combine all segments as separate columns, aligned on time
        combined_df = pd.concat(segments_dict.values(), axis=1)
        combined_df.columns = list(segments_dict.keys())
        combined_df = combined_df.sort_index()

        return combined_df


    def save_individual_station_plots(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for station, series in self.station_data.items():
            plt.figure(figsize=(15, 4))
            plt.plot(series.index, series.values, '.', color='black', markersize=0.2)
            plt.title(f"{station.upper()} - Merged Time Series")
            plt.xlabel("Time")
            plt.ylabel("Sample")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{station}_merged_plot.png"), dpi=300)
            plt.close()

    def select_samples(self, sample_indices):
        selected = []
        for station, series in self.station_data.items():
            if isinstance(sample_indices, dict):
                indices = sample_indices.get(station, [])
            else:
                indices = sample_indices
            selected_series = series.iloc[indices]
            selected.append(selected_series.rename(station))
        return pd.concat(selected, axis=1, join='inner')

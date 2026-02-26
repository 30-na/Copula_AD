import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD"

# Reads a CSV time series file, cleans it, and sets the datetime index.
def read_time_series(file_name):
    eq_raw = pd.read_csv(
    os.path.join(path, file_name),
    skiprows=18,
    parse_dates=["Time"],
    low_memory=False
    )

    eq_raw.rename(columns={" Sample": "Sample"}, inplace=True)
    eq_raw['Sample'] = pd.to_numeric(eq_raw['Sample'], errors='coerce')

    # Convert "Time" with invalid rows becoming NaT
    eq_raw["Time"] = pd.to_datetime(eq_raw["Time"], errors='coerce')

    # Drop rows with NaT in the "Time" column
    eq_raw = eq_raw.dropna(subset=["Time"])
    eq_raw["Time"] = pd.to_datetime(eq_raw["Time"])
    eq_raw = eq_raw.set_index("Time")
    return eq_raw

# Splits the time series into continuous segments based on non-monotonic timestamps.
def break_time_series(eq_raw):
    segments = []
    segment_start_idx = 0

    for i in range(1, len(eq_raw)):
        if  eq_raw.index[i-1] >= eq_raw.index[i]:
            # Break the data when the condition is met
            segments.append(eq_raw[segment_start_idx:i])
            segment_start_idx = i
    
    # Append the last segment
    segments.append(eq_raw[segment_start_idx:])
    
    return segments

# Plots each segment of the time series as a subplot and saves the figure.
def plot_time_series_segments(segments, path):
    num_segments = len(segments)  
    
    # Create subplots: one for each segment
    fig, axs = plt.subplots(num_segments, 1, figsize=(10, 4*num_segments))
    
    # Define the common time range for all subplots
    #start_time = pd.Timestamp('2018-08-15 0:00:00+00:00')
    #end_time = pd.Timestamp('2018-08-16 0:00:00+00:00')
    
    # If there's only one segment, axs won't be an array, so we ensure it is.
    if num_segments == 1:
        axs = [axs]
    
    for i, segment in enumerate(segments):
        axs[i].scatter(segment.index, segment['Sample'], color='b', s=1)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('m/s')
        #axs[i].legend()
        
        # Set the x-axis limit to the fixed time range
        #axs[i].set_xlim([start_time, end_time])
    
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.clf()
    plt.close()


# Combines specified segments from multiple time series files into a single sorted time series.
def build_concatenated_series(folders, segment_indices):
    all_segments = []

    for i in range(len(folders)):
        folder = folders[i]
        print(folder)
        seg_idx = segment_indices[i]

        folder_path = os.path.join(path, 'data', folder)
        
        files = []
        for f in os.listdir(folder_path):
            if f.endswith('.csv'):
                files.append(f)

        files.sort(key=lambda x: int(x[-8:-4]))  # sort by last 4 digits before .csv

        for file in tqdm(files, desc=f"Processing {file}"):
            full_path = os.path.join('data', folder, file)
            eq_raw = read_time_series(full_path)
            segments = break_time_series(eq_raw)
            all_segments.append(segments[seg_idx]['Sample'])

    combined_series = pd.concat(all_segments)
    combined_series = combined_series.sort_index()
    return combined_series


combined_series_afi =  build_concatenated_series(['afi'], [0])
combined_series_afi.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\processedDatata\afi.csv")

combined_series_funa = build_concatenated_series(['funa'], [0])
combined_series_funa.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\processedData\funa.csv")

combined_series_rao =  build_concatenated_series(['rao'], [0])
combined_series_rao.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\processedData\rao.csv")

combined_series_tara = build_concatenated_series(['tara'], [12])
combined_series_tara.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\processedData\tara.csv")


######################################## 
combined_series = build_concatenated_series(['afi', 'funa', 'rao', 'tara'], [0, 0, 0, 0])


eq_raw_afi_1516 = read_time_series(file_name = r"data\rao\fdsnws-dataselect_2025-05-30t23_25_39z_1718.csv")
eq_seperate_list_afi_1516 = break_time_series(eq_raw_afi_1516)
plot_time_series_segments(segments=eq_seperate_list_afi_1516, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\test3.png")

eq_raw_afi = read_time_series(file_name = r"data\afi\fdsnws-dataselect_2025-06-02t18_12_09z_1617.csv")
eq_seperate_list_afi = break_time_series(eq_raw_afi)
plot_time_series_segments(segments=eq_seperate_list_afi, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\afi.png")


eq_seperate_list_afi[0].to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\processedData\afi.csv")

eq_raw_tara = read_time_series(file_name = r"data\tara\fdsnws-dataselect_2025-05-30t22_50_20z_1617.csv")
eq_seperate_list_tara = break_time_series(eq_raw_tara)
plot_time_series_segments(segments=eq_seperate_list_tara, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\tara.png")


eq_raw_funa = read_time_series(file_name = r"data\funa\fdsnws-dataselect_2025-06-02t19_24_19z_1516.csv")
eq_seperate_list_funa = break_time_series(eq_raw_funa)
plot_time_series_segments(segments=eq_seperate_list_funa, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\funa.png")

eq_raw_rao = read_time_series(file_name = r"data\rao\fdsnws-dataselect_2025-05-30t23_25_57z_1819.csv")
eq_seperate_list_rao = break_time_series(eq_raw_rao)
plot_time_series_segments(segments=eq_seperate_list_rao, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\rao.png")
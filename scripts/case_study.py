import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD"

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


def plot_time_series_segments(segments, path):
    num_segments = len(segments)  
    
    # Create subplots: one for each segment
    fig, axs = plt.subplots(num_segments, 1, figsize=(10, 4*num_segments))
    
    # Define the common time range for all subplots
    start_time = pd.Timestamp('2018-08-18 12:00:00+00:00')
    end_time = pd.Timestamp('2018-08-19 6:00:00+00:00')
    
    # If there's only one segment, axs won't be an array, so we ensure it is.
    if num_segments == 1:
        axs = [axs]
    
    for i, segment in enumerate(segments):
        axs[i].scatter(segment.index, segment['Sample'], color='b', s=1)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('m/s')
        #axs[i].legend()
        
        # Set the x-axis limit to the fixed time range
        axs[i].set_xlim([start_time, end_time])
    
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.clf()
    plt.close()

eq_raw_afi = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t00_42_55z_AFI.csv")
eq_seperate_list_afi = break_time_series(eq_raw_afi)
plot_time_series_segments(segments=eq_seperate_list_afi, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\afi.png")
eq_seperate_list_afi[0].to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\processedData\afi.csv")

eq_raw_tara = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t01_39_26z_TARA.csv")
eq_seperate_list_tara = break_time_series(eq_raw_tara)
plot_time_series_segments(segments=eq_seperate_list_tara[1], path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\tara.png")

eq_raw_hnr = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t02_02_18z_HNR.csv")
eq_seperate_list_hnr = break_time_series(eq_raw_hnr)
plot_time_series_segments(segments=eq_seperate_list_hnr, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\hnr.png")

eq_raw_funa = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t02_38_12z_FUNA.csv")
eq_seperate_list_funa = break_time_series(eq_raw_funa)
plot_time_series_segments(segments=eq_seperate_list_funa, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\funa.png")

eq_raw_rao = read_time_series(file_name = r"data\fdsnws-dataselect_2024-10-22t03_05_29z_RAO.csv")
eq_seperate_list_rao = break_time_series(eq_raw_rao)
plot_time_series_segments(segments=eq_seperate_list_rao, path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\Copula_AD\figures\rao.png")


def align_with_dtw(vectors):
    """
    Align multiple vectors using DTW.

    Args:
        vectors (list of ndarray): A list of 1D arrays or vectors.

    Returns:
        list of ndarray: A list of aligned vectors.
    """
    # Choose a reference vector (e.g., the first vector)
    reference = vectors[0]

    aligned_vectors = []

    for vector in vectors:
        _, path = fastdtw(reference, vector, dist=euclidean)

        # Extract aligned indices
        aligned_ref_idx, aligned_vec_idx = zip(*path)

        # Align the vector
        aligned_vector = np.array([vector[idx] for idx in aligned_vec_idx])
        aligned_vectors.append(aligned_vector)

    return aligned_vectors

type(eq_seperate_list_funa[0]["Sample"])

# Example Usage
vector1 = eq_seperate_list_funa[0]["Sample"].values.reshape(-1,1)
vector2 = eq_seperate_list_rao[0]["Sample"].values.reshape(-1,1)
vector3 = np.sin(np.linspace(0, 10, 80)) - 0.1
vector4 = np.sin(np.linspace(0, 10, 110)) + 0.2
vector5 = np.sin(np.linspace(0, 10, 90)) - 0.2

aligned_vectors = align_with_dtw([vector1, vector2])
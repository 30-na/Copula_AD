import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import re
from tqdm import tqdm
import statsmodels.api as sm
from joblib import Parallel, delayed
from pmdarima import auto_arima


### Read and Extract the shape of each file
def read_dat_shapes(directory, n_channels=4, dtype=np.int16):
    dat_files = [f for f in os.listdir(directory) if f.endswith(".dat")]
    results = []
    for filename in tqdm(dat_files, desc="Reading .dat files"):
        filepath = os.path.join(directory, filename)
        try:
            raw_data = np.fromfile(filepath, dtype=dtype)
            data = raw_data.reshape(-1, n_channels)
            results.append((filename, data.shape))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return results

# Read and extarct the seizure time from .txt file
def extract_seizure_times_from_txt(txt_name: str) -> dict:
    txt_path = os.path.join("data/mice/RawData/", txt_name)
    seizure_dict = {}

    try:
        df = pd.read_csv(txt_path, sep='\t', skiprows=6)
        starts = df[df['Annotation'].str.strip().str.lower() == 'seizure starts']['Start Time'].tolist()
        ends = df[df['Annotation'].str.strip().str.lower() == 'seizure ends']['Start Time'].tolist()
        seizure_dict = {
            'starts': starts,
            'ends': ends
        }
    except Exception as e:
        print(f"Failed to read {txt_name}: {e}")

    return seizure_dict

# load the EEG .dat file as a DataFrame with time index
def load_eeg_slice_to_df(file_name: str, start_time_str: str = None, end_time_str: str = None) -> pd.DataFrame:
    
    n_channels = 4
    dtype = np.int16
    dat_path = os.path.join("data/mice/RawData/" + file_name)
    

    # Extract start datetime string from filename
    match = re.search(r"_TS_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})", file_name)
    if not match:
        raise ValueError(f"Could not extract timestamp from filename: {file_name}")
    timestamp_str = f"{match.group(1)} {match.group(2)}:{match.group(3)}:{match.group(4)}"
    start_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

    raw_data = np.fromfile(dat_path, dtype=dtype)
    data = raw_data.reshape(-1, n_channels)

    full_index = pd.date_range(start=start_dt, periods=data.shape[0], freq="0.5ms")
    df = pd.DataFrame(data, columns=["CH1", "CH2", "CH3", "CH4"], index=full_index)

    if start_time_str is not None:
        start_ts = pd.to_datetime(start_time_str)
    else:
        start_ts = df.index[0]

    if end_time_str is not None:
        end_ts = pd.to_datetime(end_time_str)
    else:
        end_ts = df.index[-1]

    df = df.loc[start_ts:end_ts].copy()
    return df

# plot the EEG time series data
def plot_eeg_df_with_seizures(df: pd.DataFrame, seizure_dict:dict, save_path: str, start_time_str: str = None, end_time_str: str = None):

    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    channel_names = [
        "Ipsilateral Hippocampus",
        "Ipsilateral Neocortex",
        "Contralateral Hippocampus",
        "Contralateral Neocortex"
    ]

    start_ts = pd.to_datetime(start_time_str) if start_time_str else df.index[0]
    end_ts = pd.to_datetime(end_time_str) if end_time_str else df.index[-1]

    seizure_start_list = seizure_dict["starts"]
    seizure_end_list = seizure_dict["ends"]
    
    for i, ax in enumerate(axes):
        #ax.scatter(df.index, df.iloc[:, i], s=2)
        ax.plot(df.index, df.iloc[:, i], linewidth=0.5)
        ax.set_title(channel_names[i])
        ax.set_ylabel("Voltage (int16)")
        ax.grid(True)

        for sz_start, sz_end in zip(seizure_start_list, seizure_end_list):
            sz_start_dt = pd.to_datetime(sz_start)
            sz_end_dt = pd.to_datetime(sz_end)
            if sz_end_dt < start_ts or sz_start_dt > end_ts:
                continue
            s1 = max(sz_start_dt, start_ts)
            s2 = min(sz_end_dt, end_ts)
            ax.axvspan(s1, s2, color='red', alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# Calculate the sigma2, Variance of residual noise, remaining unexplained variability after fitting the ARIMA
def compute_sigma2_values(data, window_size, overlap_size, order=None, n_jobs=-1):
    step = window_size - overlap_size
    data_array = np.asarray(data)
    num_windows = (len(data_array) - window_size) // step + 1
    
    start_indices = [i for i in range(0, num_windows * step, step)]
    windows = [data_array[i:i+window_size] for i in start_indices]
    timestamps = pd.to_datetime([data.index[i + window_size - 1] for i in start_indices])
    
    def fit_window(window):
        try:
            model = sm.tsa.ARIMA(window, order=order, enforce_stationarity=True, enforce_invertibility=True)
            result = model.fit()
            # variance parameter (sigma2) is appended at the end.
            # print(result.param_names)
            sigma2 = result.params[-1]
            return sigma2, order
        except:
            return np.nan, None

    def fit_window_auto(window):
        warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*force_all_finite.*"
        )
        try:
            model = auto_arima(
                window,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3,
                max_q=3, 
                max_order=6
            )
            sigma2 = model.params()[-1]
            return sigma2, model.order
             
        except:
            return np.nan, None

    if order is not None:
        result = Parallel(n_jobs=n_jobs)(
            delayed(fit_window)(w) for w in tqdm(windows, desc=f"Computing σ² for {ch}")
            )
    else:
        result = Parallel(n_jobs=n_jobs)(
            delayed(fit_window_auto)(w) for w in tqdm(windows, desc=f"Computing σ² for {ch}")
            )

    arr = np.array(result, dtype=object)

    sigma2_values = arr[:, 0]
    order_values  = arr[:, 1]


    return pd.Series(sigma2_values, index=timestamps), pd.Series(order_values, index=timestamps)


dat_file = "AC75a-5_DOB_072519_TS_2020-03-25_17_30_04_allCh.dat"
txt_key = "AC75a-5_DOB 072519_TS_2020-03-25_17_30_04.txt"

# Parameters
window_size = 2000  # 1 second for 2000 Hz
overlap_size = 500

egg_df = load_eeg_slice_to_df(dat_file)
seizure_dict = extract_seizure_times_from_txt(txt_key)
# plot_eeg_df_with_seizures(egg_df, seizure_dict, save_path="figures/March25.png")

# Apply VAD to all 4 channels
sigma2_df = pd.DataFrame()
for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
    sigma2_df[ch] = compute_sigma2_values(data=egg_df[ch], window_size=window_size, overlap_size=overlap_size)


plt.scatter(sigma2_df.index, sigma2_df["CH1"], s=1)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from ripser import ripser
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
import persim
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform, cosine, correlation
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.cm as cm
import matplotlib.dates as mdates
from scipy.stats import rankdata, norm, multivariate_normal



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

shapes = read_dat_shapes("data/mice")
for fname, shape in shapes:
    print(f"{fname}: shape = {shape}")



# Read and extarct the seizure time
def extract_seizure_times_from_txt(txt_dir):
    seizure_times = {}
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for fname in tqdm(txt_files, desc="Reading .txt annotations"):
        if "2020-03-23" in fname:
            continue  # Skip unmatched .dat file

        txt_path = os.path.join(txt_dir, fname)
        try:
            df = pd.read_csv(txt_path, sep='\t', skiprows=6)
            starts = df[df['Annotation'].str.strip().str.lower() == 'seizure starts']['Start Time'].tolist()
            ends = df[df['Annotation'].str.strip().str.lower() == 'seizure ends']['Start Time'].tolist()
            seizure_times[fname] = {
                'starts': starts,
                'ends': ends
            }
        except Exception as e:
            print(f"Failed to read {fname}: {e}")

    return seizure_times

seizure_dict = extract_seizure_times_from_txt("data/mice/train")

for fname, times in seizure_dict.items():
    print(fname)
    print("Starts:", times['starts'])
    print("Ends:  ", times['ends'])
    print()


# Plot each .dat file 
def plot_eeg_sliced_with_seizures(
    dat_path: str,
    start_time_str: str,
    end_time_str: str,
    seizure_start_list: list,
    seizure_end_list: list,
    save_path,
    s,
    start_datetime_str: str = "2020-03-24 17:30:04"
):
    n_channels = 4
    dtype = np.int16

    # Load binary EEG data
    raw_data = np.fromfile(dat_path, dtype=dtype)
    data = raw_data.reshape(-1, n_channels)

    # Build time index
    
    start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    full_index = pd.date_range(start=start_dt, periods=data.shape[0], freq = "0.5ms")
    df = pd.DataFrame(data, columns=["CH1", "CH2", "CH3", "CH4"], index=full_index)

    # Slice data
    start_ts = pd.to_datetime(start_time_str)
    end_ts = pd.to_datetime(end_time_str)
    df_slice = df.loc[start_ts:end_ts]

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    channel_names = [
        "Ipsilateral Hippocampus",
        "Ipsilateral Neocortex",
        "Contralateral Hippocampus",
        "Contralateral Neocortex"
    ]

    for i, ax in enumerate(axes):
        ax.scatter(df_slice.index, df_slice.iloc[:, i], s=s)
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
    return df_slice


def plot_all_eeg_datasets(dat_dir, output_dir, seizure_dict):
    dat_files = sorted([
        f for f in os.listdir(dat_dir)
        if f.endswith(".dat") and "2020-03-23" not in f
    ])

    for dat_file in tqdm(dat_files, desc="Plotting EEG files"):
        dat_path = os.path.join(dat_dir, dat_file)
        base_name = dat_file.replace(".dat", "")
        txt_key = base_name.replace("_DOB_", "_DOB ").replace("_allCh", ".txt")
        # Extract date from filename
        try:
            date_str = base_name.split("TS_")[1][:10]  # e.g., '2020-03-24'
            start_datetime_str = f"{date_str} 17:30:04"
            end_datetime_str = (datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S") +
                                pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Failed to parse date for {dat_file}: {e}")
            continue

        seizure_starts = seizure_dict.get(txt_key, {}).get("starts", [])
        seizure_ends = seizure_dict.get(txt_key, {}).get("ends", [])

        try:
            df = plot_eeg_sliced_with_seizures(
                dat_path=dat_path,
                start_time_str=start_datetime_str,
                end_time_str=end_datetime_str,
                seizure_start_list=seizure_starts,
                seizure_end_list=seizure_ends,
                save_path=os.path.join(output_dir, base_name + ".png"),
                start_datetime_str=start_datetime_str
            )
        except Exception as e:
            print(f"Failed to plot {dat_file}: {e}")


plot_all_eeg_datasets("data/mice", "figures", seizure_dict)        


txt_key = "AC75a-5_DOB 072519_TS_2020-03-25_17_30_04.txt"
seizure_start_list=seizure_dict[txt_key]["starts"]
seizure_end_list=seizure_dict[txt_key]["ends"]


df_slice = plot_eeg_sliced_with_seizures(
    dat_path=r"data\mice\train\AC75a-5_DOB_072519_TS_2020-03-25_17_30_04_allCh.dat",
    start_time_str="2020-03-25 17:30:04",
    end_time_str="2020-03-26 17:30:04",
    save_path=r"figures\test",
    seizure_start_list=seizure_dict[txt_key]["starts"],
    seizure_end_list=seizure_dict[txt_key]["ends"],
    start_datetime_str = "2020-03-25 17:30:04",
    s=.2
)

###############

#df_slice.index.name = 'UtcTime'
#long_df_raw_fill_mice = transform_to_long_format_rawdata(df_slice)
#mtad_eucl_fill_30_mice = mtad_analysis_by_sample(df_long=long_df_raw_fill_mice, window_size=120, distance="Euclidean")


#mtad_eucl_fill_30_mice.plot()
#plt.show()
##########################

### plot Zoom siezure
def plot_all_seizures_with_context(
    dat_path: str,
    seizure_dict: dict,
    timebefore: int,
    timeafter: int,
    save_path: str,
    start_datetime_str: str,
    s 
):
    os.makedirs(save_path, exist_ok=True)

    for txt_key, seizure_entry in seizure_dict.items():
        starts = seizure_entry["starts"]
        ends = seizure_entry["ends"]

        for i, (start_str, end_str) in enumerate(zip(starts, ends)):
            seizure_start_dt = pd.to_datetime(start_str)
            seizure_end_dt = pd.to_datetime(end_str)

            start_time_str = (seizure_start_dt - pd.Timedelta(seconds=timebefore)).strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = (seizure_end_dt + pd.Timedelta(seconds=timeafter)).strftime("%Y-%m-%d %H:%M:%S")

            save_name = f"{os.path.splitext(txt_key)[0]}_seizure_{i:02d}.png"
            full_save_path = os.path.join(save_path, save_name)

            plot_eeg_sliced_with_seizures(
                dat_path=dat_path,
                start_time_str=start_time_str,
                end_time_str=end_time_str,
                seizure_start_list=[start_str],
                seizure_end_list=[end_str],
                save_path=full_save_path,
                s=s,
                start_datetime_str=start_datetime_str
            )


filtered_dict = {
    "AC75a-5_DOB 072519_TS_2020-03-28_17_30_04.txt": seizure_dict["AC75a-5_DOB 072519_TS_2020-03-28_17_30_04.txt"]
}

plot_all_seizures_with_context(
    dat_path="data/mice/AC75a-5_DOB_072519_TS_2020-03-28_17_30_04_allCh.dat",
    seizure_dict=filtered_dict,
    timebefore=180,
    timeafter=30,
    save_path="figures/march28_3600",
    start_datetime_str="2020-03-28 17:30:04",
    s=.2
)




## APPLY Sigma2
# Parameters
window_size = 20000  # 10 second for 2000 Hz
overlap_size = 5000
order = (1, 1, 1)

# Apply to all 4 channels
sigma2_df = pd.DataFrame()
for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
    sigma2_df[ch] = compute_sigma2_values_timestamps(df_slice[ch], window_size, overlap_size, order)

sigma2_df_auto = pd.DataFrame()
for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
    sigma2_df_auto[ch] = compute_sigma2_values_timestamps_autoArima(df_slice[ch], window_size, overlap_size)


sigma2_df.to_csv("processedData/sigma2_AC75a-5_DOB_072519_TS_2020-03-25_111_10SecWin_01.csv")
sigma2_df_1sec_March25 = pd.read_csv("processedData/sigma2_AC75a-5_DOB_072519_TS_2020-03-25_111_10SecWin_01.csv", index_col=0, parse_dates=True)

#################################

def compute_joint_survival_gaussian(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    U = np.array([rankdata(df[col]) / (n + 1) for col in df.columns]).T
    # U = np.clip(U, 1e-12, 1 - 1e-12)
    Z = norm.ppf(U)
    corr_matrix = np.corrcoef(Z, rowvar=False)
    copula = multivariate_normal(mean=np.zeros(Z.shape[1]), cov=corr_matrix)
    surv_vals = np.array([copula.cdf(-z) for z in tqdm(Z, desc="Computing joint survival")])
    return pd.Series(surv_vals, index=df.index, name="joint_survival")

# training helper 
def train_gaussian_copula(df: pd.DataFrame):
    n = len(df)
    U = np.array([rankdata(df[c]) / (n + 1) for c in df.columns]).T
    Z = norm.ppf(U)
    corr = np.corrcoef(Z, rowvar=False)          # Sigma
    ref = {c: np.sort(df[c].to_numpy()) for c in df.columns}  # for empirical CDF lookups
    return {"corr": corr, "ref": ref, "n": n, "cols": list(df.columns)}

# scoring with the trained model 
def compute_joint_survival_gaussian_with_model(df: pd.DataFrame, model) -> pd.Series:
    n = model["n"]
    cols = model["cols"]
    U = np.column_stack([
        (np.searchsorted(model["ref"][c], df[c].to_numpy(), side="right") + 1) / (n + 1)
        for c in cols
    ])
    # U = np.clip(U, 1e-12, 1 - 1e-12)  # optional, same as your style
    Z = norm.ppf(U)
    copula = multivariate_normal(mean=np.zeros(Z.shape[1]), cov=model["corr"])
    surv_vals = np.array([copula.cdf(-z) for z in tqdm(Z, desc="Computing joint survival (OOS)")])
    return pd.Series(surv_vals, index=df.index, name="joint_survival")


# Select training window by time
train_start = "2020-03-28 18:05:00"
train_end   = "2020-03-29 00:00:00"

# train_start = "2020-03-29 00:00:00"
# train_end   = "2020-03-29 08:00:00"

df_train = sigma2_df_1sec.loc[train_start:train_end]

# Train the Gaussian copula model
model = train_gaussian_copula(sigma2_df_1sec_March25)

# Score the whole dataset out-of-sample using trained model
joint_surv_all_01 = compute_joint_survival_gaussian_with_model(sigma2_df_1sec_March25, model)
joint_surv_all_01.to_csv("processedData/joint_surv_all_01_1sec_train_all.csv")


# def compute_joint_cdf_gaussian(df: pd.DataFrame) -> pd.Series:
#     n = len(df)

#     # Step 1: Convert to uniform marginals using empirical CDF
#     U = np.array([rankdata(df[col]) / (n + 1) for col in df.columns]).T

#     # Step 2: Transform to standard normal space
#     Z = norm.ppf(U)

#     # Step 3: Estimate correlation matrix
#     corr_matrix = np.corrcoef(Z, rowvar=False)

#     # Step 4: Compute Gaussian copula joint CDF with progress bar
#     copula = multivariate_normal(mean=np.zeros(Z.shape[1]), cov=corr_matrix)
#     joint_cdf_vals = np.array([copula.cdf(z) for z in tqdm(Z, desc="Computing joint CDF")])

#     return pd.Series(joint_cdf_vals, index=df.index, name="joint_cdf")



# joint_cdf_series_survival_10sec = compute_joint_survival_gaussian(sigma2_df)
# joint_cdf_survival_1sec = compute_joint_survival_gaussian(sigma2_df_1sec)

# joint_cdf_ma10 = joint_cdf_series_survival.rolling(window=10, min_periods=1).mean()
# joint_cdf_ma10 = joint_surv_all.rolling(window=10, min_periods=1).mean()

# joint_cdf = joint_cdf_series_survival_10sec



# joint_cdf_series_gumbel = compute_joint_cdf_gumbel_multivariate(sigma2_df)
# cdf_diff = joint_cdf_series_gumbel.diff()



seizure_start_list = [pd.to_datetime(t) for t in seizure_start_list]
seizure_end_list = [pd.to_datetime(t) for t in seizure_end_list]



#joint_cdf_series_gaussian = compute_joint_cdf_gaussian(sigma2_df)
#joint_cdf_series_Tdistribution = compute_joint_cdf_tcopula(sigma2_df)
#joint_cdf_series_gumbel = compute_joint_cdf_gumbel_multivariate(sigma2_df)
#joint_cdf = joint_cdf_series_gumbel.diff()
#joint_cdf_series_gumbel_raw = compute_joint_cdf_gumbel_multivariate(df_slice)




########################
#joint_cdf_tmp = joint_surv_all_01.rolling(window=20, min_periods=1).mean()
joint_cdf_tmp = joint_surv_all_01.copy()
sigma2_df = sigma2_df_1sec_March25.copy()
# Plotting
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

channel_titles = [
    "Ipsilateral Hippocampus",
    "Ipsilateral Neocortex",
    "Contralateral Hippocampus",
    "Contralateral Neocortex",
    "Joint CDF"
]

# Compute global x-limits from all data
# x_min = min(sigma2_df.index.min(), joint_cdf_tmp.index.min())
# x_max = max(sigma2_df.index.max(), joint_cdf_tmp.index.max())
# x_min = pd.Timestamp(seizure_start_list[7]-pd.Timedelta(minutes=8))
# x_max = pd.Timestamp(seizure_end_list[7]+pd.Timedelta(minutes=2))
x_min = pd.Timestamp(sigma2_df.index.min())
x_max = pd.Timestamp(sigma2_df.index.max())
#x_min = pd.Timestamp("2020-03-29 11:00:00")
#x_max = pd.Timestamp("2020-03-29 13:00:00")
# Compute global y-limits for σ² plots
y_min = sigma2_df.min().min()
y_max = sigma2_df.max().max()

# Plot σ² for each channel
for i, ch in enumerate(sigma2_df.columns):
    axes[i].scatter(sigma2_df.index, sigma2_df[ch], color='black', s=.2)
    axes[i].set_title(channel_titles[i])
    axes[i].set_ylabel("σ²")
    axes[i].grid(False)
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    for sz_start, sz_end in zip(seizure_start_list, seizure_end_list):
        s1 = pd.to_datetime(sz_start)
        s2 = pd.to_datetime(sz_end)
        axes[i].axvspan(s1, s2, color='red', alpha=0.2)

# Plot joint CDF
threshold = 0.0001 # set your desired threshold

# Plot joint CDF with threshold coloring
below_thresh = joint_cdf_tmp < threshold
above_thresh = ~below_thresh

axes[4].scatter(joint_cdf_tmp.index[above_thresh],
                joint_cdf_tmp[above_thresh],
                s=.2, color='black')
axes[4].scatter(joint_cdf_tmp.index[below_thresh],
                joint_cdf_tmp[below_thresh],
                s=7, color='red')

axes[4].set_title(channel_titles[4])
axes[4].set_ylabel("CDF")
axes[4].grid(False)
axes[4].set_xlim(x_min, x_max)

# Shade seizures
for sz_start, sz_end in zip(seizure_start_list, seizure_end_list):
    s1 = pd.to_datetime(sz_start)
    s2 = pd.to_datetime(sz_end)
    axes[4].axvspan(s1, s2, color='red', alpha=0.2)


axes[-1].set_xlabel("Time")
plt.tight_layout()
#plt.savefig("figures/jointcdf_AC75a-TS_2020-03-28_sigma2_111_survival_1secWin_01_mv20.png", dpi=300)
plt.show()
plt.close()






#############################
import os
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
import persim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform, cosine, correlation
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.cm as cm
import matplotlib.dates as mdates



fault_times = pd.DatetimeIndex(['2023-09-15 21:41:09', '2023-09-15 21:41:11', '2023-09-15 21:41:11', 
                            '2023-12-11 08:19:05', '2023-12-11 14:54:17', '2023-12-11 14:54:18', 
                            '2023-12-11 14:54:25', '2023-12-11 14:54:54', '2023-12-11 19:35:26', 
                            '2023-12-11 21:02:37', '2023-12-11 23:58:31', '2023-12-11 23:59:14', 
                            '2023-12-12 00:00:22', '2024-01-17 20:20:34', '2023-07-01 15:10:34', 
                            '2023-08-04 19:23:11', '2023-08-11 01:54:03', '2023-08-26 15:38:41', 
                            '2023-09-13 17:40:58', '2023-09-16 09:26:18', '2023-09-27 19:01:17', 
                            '2023-10-28 18:37:35', '2023-12-04 16:57:40', '2024-01-18 15:11:46', 
                            '2024-02-25 06:19:51', '2024-02-25 21:14:15', '2024-02-25 21:15:27', 
                            '2024-03-26 11:11:17', '2023-09-11 14:42:50', '2023-09-24 04:22:49', 
                            '2023-10-07 16:20:01', '2023-10-15 04:19:57', '2023-10-30 13:06:15', 
                            '2023-11-20 20:35:34', '2023-11-27 14:07:45', '2023-12-11 08:19:05', 
                            '2023-12-11 08:25:25', '2023-12-11 14:54:01', '2023-12-11 14:54:25', 
                            '2023-12-11 14:54:54', '2023-12-11 19:35:26', '2023-12-11 21:02:37', 
                            '2023-12-11 23:58:31', '2023-12-11 23:59:14', '2023-12-12 00:00:22', 
                            '2024-01-14 23:57:39', '2024-02-01 16:35:48', '2024-02-02 20:26:29', 
                            '2024-02-05 14:56:55', '2024-02-19 00:43:32', '2024-03-16 15:42:23', 
                            '2023-07-09 13:43:37', '2023-07-10 17:37:45', '2023-07-20 16:48:10', 
                            '2023-08-24 20:01:05', '2023-08-31 18:37:07', '2023-09-06 07:28:31', 
                            '2023-09-07 21:55:36', '2023-09-15 21:40:55', '2023-09-16 09:49:05', 
                            '2023-09-26 16:55:41', '2023-10-08 08:07:30', '2023-10-20 20:03:39', 
                            '2023-11-23 19:31:03', '2023-12-12 20:43:40', '2023-12-12 20:49:54', 
                            '2023-12-12 20:50:52', '2023-12-12 20:51:16', '2023-12-12 20:51:34', 
                            '2023-12-13 16:39:28', '2023-12-19 16:55:58', '2024-02-25 15:50:44', 
                            '2024-02-29 21:27:57', '2024-03-05 06:48:56', '2023-08-30 15:33:30', 
                            '2023-09-08 04:51:42', '2023-09-25 16:39:33', '2023-11-12 04:16:37', 
                            '2023-11-12 16:50:24', '2023-11-30 21:00:29', '2023-12-01 17:58:44', 
                            '2023-12-01 20:49:59'])



def read_and_merge_csv(folder_path):

    all_data = []
    column_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            if 'UtcTime' in df.columns and 'prediction' in df.columns:
                df['UtcTime'] = pd.to_datetime(df['UtcTime'], errors='coerce')
                df.set_index('UtcTime', inplace=True)
                all_data.append(df['prediction'])
                column_names.append(os.path.splitext(file)[0])
                print(f"{file} added")
            else:
                print(f"Skipped {file} as it doesn't contain required columns.")

    merged_df = pd.concat(all_data, axis=1)
    merged_df.columns = column_names
    merged_df = merged_df.apply(pd.to_numeric, errors="coerce")
    merged_df.to_csv(os.path.join(folder_path, r"merged_data\mergedOutput.csv"))
    print(f"Data merged and saved to {folder_path}")
    return(merged_df)


def variable_selection(df):
    category_dict = {}
    for col in df.columns:
        category, variable = col.split('%')
        category = category.split('_')[-1]

        if category not in ["Radio", "Refs"]:  # Exclude "Radio" and "Refs" due to the raw data stop updating after 2023-09-25(Radio) and 2023-12-22(Refs)
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(variable)

    return category_dict


def read_file_category_XVI(category, time_index='UtcTime'):
    # Read the CSV file and make a new dataframe
    path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\XVIData"
    file_name = category + ".csv"
    file = pd.read_csv(os.path.join(path, file_name))
    file[time_index] = pd.to_datetime(file[time_index], format='ISO8601')
    file = file.set_index(time_index)
    return file


def raw_input_data_fill_gap():
    satsam_file = read_and_merge_csv(folder_path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\outputArchive")
    satsam_variables_dict = variable_selection(df=satsam_file)
    combined_df = pd.DataFrame()

    for category, variables in tqdm(satsam_variables_dict.items()):
        file = read_file_category_XVI(category)  
        selected_columns = [col for col in file.columns if col in variables]

        # Subset and rename columns
        subset_df = file[selected_columns]
        subset_df.columns = [f"{category}_{col}" for col in selected_columns]

        # cleaning and standardize
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(scaler.fit_transform(subset_df), columns=subset_df.columns)
        df_standardized.index = subset_df.index
    
        # resampling and interpolation
        resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample("1min").mean().interpolate(method="linear")
       
        # Combine into final DataFrame
        combined_df = pd.concat([combined_df, resample], axis=1)
        
    return combined_df


def raw_input_data_ignore_gap():
    satsam_file = read_and_merge_csv(folder_path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\outputArchive")
    satsam_variables_dict = variable_selection(df=satsam_file)
    combined_df = pd.DataFrame()

    for category, variables in tqdm(satsam_variables_dict.items()):
        file = read_file_category_XVI(category)  
        selected_columns = [col for col in file.columns if col in variables]
        subset_df = file[selected_columns]
        subset_df.columns = [f"{category}_{col}" for col in selected_columns]

        scaler = StandardScaler()
        df_standardized = pd.DataFrame(scaler.fit_transform(subset_df), columns=subset_df.columns)
        df_standardized.index = subset_df.index

        resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample("1min").mean()
        resample = resample.dropna()

        combined_df = pd.concat([combined_df, resample], axis=1)

    return combined_df


def transform_to_long_format_rawdata(data):
    long_data = (data.reset_index().melt(id_vars='UtcTime', var_name="variable", value_name="value"))
    long_data.rename(columns={"UtcTime": "time"}, inplace=True)
    # add a column name model and assigne "rawdata"
    long_data["layer"] = "rawdata"
    long_data['time'] = pd.to_datetime(long_data['time'])
    return long_data


def compute_distance_matrix(window_numArray, distance, threshold=None):
    
    if distance == "Hamming":
        dist_matrix = squareform(pdist(window_numArray, metric=lambda a, b: np.sum(a != b)))

    elif distance == "Euclidean":
        dist_matrix = squareform(pdist(window_numArray, metric='euclidean'))

    elif distance == "Jaccard":
        window_numArray = window_numArray.astype(bool)
        dist_matrix = squareform(pdist(window_numArray, metric='jaccard'))
    
    elif distance == "Manhattan":
        dist_matrix = squareform(pdist(window_numArray, metric='cityblock'))
    
    elif distance == "Cosine":
        dist_matrix = squareform(pdist(window_numArray, metric='cosine'))
    
    elif distance == "Correlation":
        # Add very small noise to rows with zero standard deviation
        stds = np.std(window_numArray, axis=1)
        zero_std_mask = stds == 0

        if np.any(zero_std_mask):
            noise = np.random.normal(loc=0, scale=1e-8, size=window_numArray[zero_std_mask].shape)
            window_numArray[zero_std_mask] += noise

        dist_matrix =  1 - abs(np.corrcoef(window_numArray))

    else:
        raise ValueError("Undefined Distance Metric")
    
    if threshold is not None:
        dist_vector = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        cutoff = np.percentile(dist_vector, threshold * 100)
        dist_matrix[dist_matrix > cutoff] = 0

    return dist_matrix


def create_graphs(data, window_size, nodes, layer, distance, threshold=None, n_jobs=-1):
    layers = data[layer].unique()
    temporal_graphs = {}

    for l in tqdm(layers):
        layer_data = data[data[layer] == l].copy()
        layer_data = layer_data.pivot_table(index='time', columns=nodes, values='value')
        layer_data = layer_data.sort_index()

        windows = []
        timestamps = []

        for start in range(0, len(layer_data), window_size):
            end = start + window_size
            window = layer_data.iloc[start:end]
            if len(window) < window_size:
                continue  # skip incomplete window
            windows.append(window.T)  # transpose: rows = variables
            timestamps.append(window.index[0])  # first time in window

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_distance_matrix)(window_T.values, distance, threshold)
            for window_T in windows
        )

        temporal_graphs[l] = {
            time: adj for time, adj in zip(timestamps, results) if adj is not None
        }

    return temporal_graphs


def compute_geodesic_distances_scipy(temporal_graphs):
    
    geodesic_distances = {}

    for model, time_graphs in temporal_graphs.items():
        model_distances = {}

        for time_window, adjacency_matrix in time_graphs.items():
            # Compute geodesic distance matrix using scipy
            distance_matrix = shortest_path(csgraph=adjacency_matrix, directed=False, unweighted=False)
            model_distances[time_window] = distance_matrix

        geodesic_distances[model] = model_distances

    return geodesic_distances


def compute_persistence_diagrams(geodesic_distances):

    """
    Computes persistence diagrams from geodesic distance matrices using Ripser.

    Parameters:
    geodesic_distances (dict): A nested dictionary {Layer -> {time_window -> distance_matrix}}.

    Returns:
    dict: A nested dictionary {Layer -> {time_window -> persistence_diagram}}.

    """
    persistence_diagrams = {}

    for layer, time_windows in geodesic_distances.items():
        layer_diagrams = {}

        for time_window, distance_matrix in time_windows.items():
            # Ensure the distance matrix is a NumPy array
            distance_matrix = np.array(distance_matrix)

            # Compute persistence diagram using Ripser
            result = ripser(distance_matrix, distance_matrix=True)
            diagram = result['dgms']

            layer_diagrams[time_window] = diagram

        persistence_diagrams[layer] = layer_diagrams

    return persistence_diagrams


def stack_persistence_diagrams_by_time(persistence_diagrams):
    """
    Stacks persistence diagrams across time, assigning layer IDs for each model and dimension.

    Parameters:
    persistence_diagrams (dict): A nested dictionary {Layer -> {time_stamp -> persistence_diagram}}.

    Returns:
    dict: A dictionary {time_stamp -> stacked_persistence_diagram} where each persistence diagram is 
          an Nx3 NumPy array with (layer_id, birth, death).

    Notes:
    - Assigns a unique layer ID for each (Layer, dimension) pair.
    """
    
    # Initialize a dictionary to store stacked persistence diagrams
    stacked_diagrams = {}

    # Get all unique time stamps from the input
    time_stamps = set()
    for model in persistence_diagrams:
        time_stamps.update(persistence_diagrams[model].keys())

    # Sort the time stamps for consistency
    time_stamps = sorted(time_stamps)

    # Create a mapping from (model_id, dimension) to layer ID
    layer_mapping = {}
    layer_counter = 0

    for model_id, (model_name, model_pds) in enumerate(persistence_diagrams.items()):
        for dim in range(len(next(iter(model_pds.values()), []))):  # Assume all time stamps have the same dimensions
            layer_mapping[(model_id, dim)] = layer_counter
            layer_counter += 1

    # Process each time stamp
    for time_stamp in time_stamps:
        spd = []  # Temporary list to store SPD for this time stamp

        # Loop through models to stack their PDs for the current time stamp
        for model_id, (model_name, model_pds) in enumerate(persistence_diagrams.items()):
            if time_stamp in model_pds:
                # Get the persistence diagram for this model at this time stamp
                pd = model_pds[time_stamp]
                # Add layer ID (calculated from model_id and dim) and birth-death information
                for dim, features in enumerate(pd):
                    layer_id = layer_mapping[(model_id, dim)]
                    for birth, death in features:
                        spd.append((layer_id, birth, death))

        # Convert the stacked list to a NumPy array
        stacked_diagrams[time_stamp] = np.array(spd)

    return stacked_diagrams


def compute_wasserstein_distances(stacked_diagrams):
    """
    Computes the Wasserstein distance for each layerID separately and sums them up.
    
    :param stacked_diagrams: Dictionary {timestamp: persistence diagram}
                             Each persistence diagram is an Nx3 array with [layerID, birth, death].
    :return: (original_distances, normalized_distances) - both as lists
    """

    # Sort time stamps for consecutive comparisons
    time_stamps = sorted(stacked_diagrams.keys())
    wasserstein_distances = []

    # Loop through consecutive time stamps
    for i in range(len(time_stamps) - 1):
        t1, t2 = time_stamps[i], time_stamps[i + 1]
        pd1, pd2 = stacked_diagrams[t1], stacked_diagrams[t2]

        # Get unique layers
        layers = np.unique(np.concatenate([pd1[:, 0], pd2[:, 0]]))

        total_distance = 0
        for layer in layers:
            # Filter persistence diagrams by layer
            pd1_layer = pd1[pd1[:, 0] == layer][:, 1:]
            pd2_layer = pd2[pd2[:, 0] == layer][:, 1:]

            # Remove points with infinite death times (there may be a better solution for this!!!)
            pd1_layer = pd1_layer[np.isfinite(pd1_layer[:, 1])]
            pd2_layer = pd2_layer[np.isfinite(pd2_layer[:, 1])]

            # Compute Wasserstein distance for this layer
            layer_distance = persim.wasserstein(pd1_layer, pd2_layer)
            total_distance += layer_distance  # Sum up all layer distances

        wasserstein_distances.append(total_distance)
    time_stamps = time_stamps[1:]
    # Normalize distances
    #max_distance = max(wasserstein_distances)
    #normalized_distances = [dist / max_distance for dist in wasserstein_distances]

    return time_stamps, wasserstein_distances


def plot_mtad_distances(df):
    df['Time'] = pd.to_datetime(df['Time'])
    plt.figure(figsize=(10, 4))
    plt.scatter(df['Time'], df['Distance'], s=10)  # s controls marker size
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.title('MTAD Distances Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def plot_wasserstein_time_series_flag_anomalies(time_stamps, wasserstein_distances, k=3, reset=None, title="MTAD Analysis"):
    # Convert timestamps to pandas datetime
    time_series_df = pd.DataFrame({"Time": time_stamps, "Distance": wasserstein_distances})
    # save the time_series_df to csv file

    # flag the anomalies
    time_series_df['Anomaly'] = time_series_df['Distance'] > time_series_df['Distance'].mean() + k*time_series_df['Distance'].std()
    anomalies = time_series_df[time_series_df['Anomaly']]


    # Plot the time series
    plt.figure(figsize=(14, 6))
    plt.plot(time_series_df['Time'], time_series_df['Distance'], color="blue", linestyle="-", marker="o", markersize=1 
             #label='WTDA'
             )
    plt.scatter(anomalies['Time'], anomalies['Distance'], color='red',
                 #label='Anomalies', 
                 marker='x')
    
    if reset is not None:
        for rt in reset:
            plt.axvline(rt, color='r', alpha=.3, linestyle='--', lw=2, label=f'Reset')

    plt.xlabel("Time")
    plt.ylabel("")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    #plt.legend()
    plt.show()
    # Save plot
    #plt.savefig(fr"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\performance\{file_name}.png")
    plt.close()
    
    return time_series_df


def mtad_analysis_by_sample(df_long, window_size, distance, nodes='variable', layer='layer', graph_threshold=None):
    graphs = create_graphs(data=df_long, window_size=window_size, nodes=nodes, layer=layer, distance=distance, threshold=graph_threshold, n_jobs=-1)
    geodesic_distances = compute_geodesic_distances_scipy(graphs)
    persistence_diagrams = compute_persistence_diagrams(geodesic_distances)
    stacked_diagrams = stack_persistence_diagrams_by_time(persistence_diagrams)
    time_stamps, distances = compute_wasserstein_distances(stacked_diagrams)
    distances_df = pd.DataFrame({'Time': time_stamps, 'Distance': distances}).set_index('Time')
    return distances_df
##########################




###############################
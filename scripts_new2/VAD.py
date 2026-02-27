import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import scipy.stats as stats
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed



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
    #merged_df.to_csv(os.path.join(folder_path, r"merged_data\mergedOutput.csv"))
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


def compute_sigma2_values_timestamps(data, window_size, overlap_size, order):
    def create_windows(data_array, window_size, overlap_size):
        step = window_size - overlap_size
        num_windows = (len(data_array) - window_size) // step + 1
        start_indices = [i for i in range(0, num_windows * step, step)]
        windows = np.array([data_array[i:i+window_size] for i in start_indices])
        end_timestamps = [data.index[i + window_size - 1] for i in start_indices]
        return windows, pd.to_datetime(end_timestamps)

    def fit_arima_sigma2(windows, order):
        sigma2_values = []
        for window in windows:
            model = sm.tsa.ARIMA(window, order=order)
            result = model.fit()
            sigma2_values.append(result.params[-1])
        return np.array(sigma2_values)
    
    # def fit_arima_sigma2(windows, order):
    #     def fit_one(window):
    #         return sm.tsa.ARIMA(window, order=order).fit().params[-1]

    #     return np.array(Parallel(n_jobs=-1)(delayed(fit_one)(w) for w in windows))

    data = data.copy()
    windows, timestamps = create_windows(np.asarray(data), window_size, overlap_size)
    sigma2_values = fit_arima_sigma2(windows, order)

    return pd.Series(sigma2_values, index=timestamps)


def compute_sigma2_all_columns(df, window_size, overlap_size, order):
    result_dict = {}
    all_timestamps = None

    for col in tqdm(df.columns, desc="Processing columns"):
        series = df[col]
        sigma2_series = compute_sigma2_values_timestamps(series, window_size, overlap_size, order)

        col_name = "VAD_" + col.replace("_", "%")
        result_dict[col_name] = sigma2_series

        if all_timestamps is None:
            all_timestamps = sigma2_series.index

    result_df = pd.DataFrame(result_dict, index=all_timestamps)
    return result_df


def plot_time_series_flag_anomalies(time_stamps, distances, k=3, reset=None, title="Anomaly Score Analysis"):
    # Convert timestamps to pandas datetime
    time_series_df = pd.DataFrame({"Time": time_stamps, "Distance": distances})
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
    
def flag_outliers_std_all(df, k):
    binary_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        threshold = mean + k * std
        binary_df[col] = (df[col] > threshold).astype(int)
    return binary_df


def extract_anomaly_ranges(df, column):
    ranges = []
    current_range = []

    for i, value in enumerate(df[column]):
        if value == 1:
            current_range.append(i)
        elif current_range:
            ranges.append(current_range)
            current_range = []

    if current_range:
        ranges.append(current_range)

    return ranges


def delta_flat(i, L):
    return 1


def delta_front(i, L):
    return L - i + 1


def delta_back(i, L):
    return i


def delta_middle(i, L):
    if i <= L // 2:
        return i
    else:
        return L - i + 1
    

def omega(anomaly_range, overlap_set, delta_fn):
    my_value = 0
    max_value = 0
    L = len(anomaly_range)

    for i in range(1, L + 1):  
        bias = delta_fn(i, L)
        max_value += bias
        if anomaly_range[i - 1] in overlap_set:
            my_value += bias

    return my_value / max_value if max_value > 0 else 0


def cardinality_factor(target_range, reference_ranges, gamma_fn):
    overlap_count = 0
    target_set = set(target_range)

    for ref in reference_ranges:
        if target_set.intersection(ref):
            overlap_count += 1

    if overlap_count <= 1:
        return 1.0
    else:
        return gamma_fn(overlap_count)


def overlap_reward(real_range, predicted_ranges, delta_fn, gamma_fn):
    overlap_sum = 0
    real_set = set(real_range)

    for pred in predicted_ranges:
        overlap = real_set.intersection(pred)
        if overlap:
            overlap_sum += omega(real_range, overlap, delta_fn)

    c_factor = cardinality_factor(real_range, predicted_ranges, gamma_fn)
    return c_factor * overlap_sum


def existence_reward(real_range, predicted_ranges):
    real_set = set(real_range)

    for pred in predicted_ranges:
        if real_set.intersection(pred):
            return 1.0

    return 0.0


def recall_T(real_range, predicted_ranges, alpha, delta_fn, gamma_fn):
    exist = existence_reward(real_range, predicted_ranges)
    overlap = overlap_reward(real_range, predicted_ranges, delta_fn, gamma_fn)
    return alpha * exist + (1 - alpha) * overlap


def total_recall_T(real_ranges, predicted_ranges, alpha, delta_fn, gamma_fn):
    total = 0.0
    for r in real_ranges:
        total += recall_T(r, predicted_ranges, alpha, delta_fn, gamma_fn)

    return total / len(real_ranges)


def precision_T(pred_range, real_ranges, delta_fn, gamma_fn):
    overlap_sum = 0
    pred_set = set(pred_range)

    for real_range in real_ranges:
        overlap = pred_set.intersection(real_range)
        if overlap:
            overlap_sum += omega(
                anomaly_range=pred_range,
                overlap_set=overlap,
                delta_fn=delta_fn
            )

    c_factor = cardinality_factor(
        target_range=pred_range,
        reference_ranges=real_ranges,
        gamma_fn=gamma_fn
    )

    return c_factor * overlap_sum


def total_precision_T(real_ranges, predicted_ranges, delta_fn, gamma_fn):
    if not predicted_ranges:
        return 0.0

    total = 0.0
    for pred_range in predicted_ranges:
        total += precision_T(
            pred_range=pred_range,
            real_ranges=real_ranges,
            delta_fn=delta_fn,
            gamma_fn=gamma_fn
        )

    return total / len(predicted_ranges)


def f1_T(real_ranges, predicted_ranges, alpha, delta_fn, gamma_fn):
    recall = total_recall_T(
        real_ranges=real_ranges,
        predicted_ranges=predicted_ranges,
        alpha=alpha,
        delta_fn=delta_fn,
        gamma_fn=gamma_fn
    )

    precision = total_precision_T(
        real_ranges=real_ranges,
        predicted_ranges=predicted_ranges,
        delta_fn=delta_fn,
        gamma_fn=gamma_fn
    )

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def mark_fault_intervals(df, fault_times, start=None, end=None, back_minutes=0, forward_minutes=0):
    df.index = pd.to_datetime(df.index)

    if start:
        start = pd.to_datetime(start).tz_localize("UTC")
    if end:
        end = pd.to_datetime(end).tz_localize("UTC")
    if start or end:
        df = df.loc[start:end]

    fault_times_UTC = pd.to_datetime(fault_times).tz_localize('UTC')

    # Slice fault times to same [start, end] window
    if start:
        fault_times_UTC = fault_times_UTC[fault_times_UTC >= start]
    if end:
        fault_times_UTC = fault_times_UTC[fault_times_UTC <= end]

    index = df.index
    prediction = pd.Series(0, index=index)

    for t in fault_times_UTC:
        raw_start = t - pd.Timedelta(minutes=back_minutes)
        raw_end = t + pd.Timedelta(minutes=forward_minutes)

        start_time = index[index <= raw_start][-1]
        end_time = index[index <= raw_end][-1]

        interval_mask = (index >= start_time) & (index <= end_time)
        prediction[interval_mask] = 1

    result = df.copy()
    result['Prediction'] = prediction


    return result


def evaluate_range_based_metrics(df, alpha=0.5, delta_fn=delta_front, gamma_fn=lambda x: 1 / x):
    results = []
    real_ranges = extract_anomaly_ranges(df, 'Prediction')

    for column in df.columns:
        if column == 'Prediction':
            continue

        predicted_ranges = extract_anomaly_ranges(df, column)

        recall = total_recall_T(
            real_ranges=real_ranges,
            predicted_ranges=predicted_ranges,
            alpha=alpha,
            delta_fn=delta_fn,
            gamma_fn=gamma_fn
        )

        precision = total_precision_T(
            real_ranges=real_ranges,
            predicted_ranges=predicted_ranges,
            delta_fn=delta_fn,
            gamma_fn=gamma_fn
        )

        f1 = f1_T(
            real_ranges=real_ranges,
            predicted_ranges=predicted_ranges,
            alpha=alpha,
            delta_fn=delta_fn,
            gamma_fn=gamma_fn
        )
        
        model_name, rest = column.split("_", 1)
        category_name, variable_name = rest.split("%", 1)
        
        results.append({
            "Model": model_name,
            "Category": category_name,
            "Variable": variable_name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    return pd.DataFrame(results)




df_raw_fill = raw_input_data_fill_gap()
df_raw_fill = df_raw_fill.dropna()
df_filtered = df_raw_fill[df_raw_fill.index >= '2023-07-10 17:00:00+00:00'] # to be match with SatSAM
df_sigma2 = compute_sigma2_all_columns(df=df_filtered, window_size=12*60, overlap_size=0, order=(1,1,1))
df_sigma2.to_csv("C:/Users/Sina.Mokhtar.XLSCIENTIFIC/Documents/Problems/XVI/VCD/sigma2_result_121_overlap0.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(df_sigma2.T, cmap="coolwarm", robust=True, 
            xticklabels=50, yticklabels=True)  # Transpose to have time on x-axis

plt.title("Heatmap of Sigma2 Values")
plt.xlabel("Time")
plt.ylabel("Variables")
plt.xticks(rotation=45)
plt.show()

sigma2_result_sum = df_sigma2.sum(axis=1)
df = sigma2_result_sum.to_frame(name="sigma2_all%Sum")
df_log_sigma2= np.log10(df_sigma2)

sigma2_result_sum = df_sigma2.sum(axis=1)
plot_time_series_flag_anomalies(sigma2_result_sum.index, sigma2_result_sum.values, k=3, reset=fault_times, title="Anomaly Score Analysis")

sigma2_log_sum = df_log_sigma2.sum(axis=1).diff()
plot_time_series_flag_anomalies(sigma2_log_sum.index, sigma2_log_sum.values, k=2, reset=fault_times, title="Anomaly Score Analysis")

df_log_sigma2= np.log(df_sigma2)
# df has anomaly scores in all columns
sigma2_binary = flag_outliers_std_all(df_sigma2, k=1)


df_marked = mark_fault_intervals(df=sigma2_binary, fault_times=fault_times, start='2023-08-01', end='2024-04-01', back_minutes=360, forward_minutes=0)
evaluation_df = evaluate_range_based_metrics(df=df_marked, alpha=0.5, delta_fn=delta_flat, gamma_fn=lambda x: 1 / x)
evaluation_df = evaluation_df.sort_values(by="F1", ascending=False).reset_index(drop=True)
evaluation_df
averages = evaluation_df.groupby("Model")[["Precision", "Recall", "F1"]].mean().reset_index()
averages
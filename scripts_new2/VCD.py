import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt

def read_file_category_XVI(category, time_index='UtcTime'):
    # Read the CSV file and make a new dataframe
    path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\XVIData"
    file_name = category + ".csv"
    file = pd.read_csv(os.path.join(path, file_name))
    file[time_index] = pd.to_datetime(file[time_index], format='ISO8601')
    file = file.set_index(time_index)
    return file


def process_category(category):
    file = read_file_category_XVI(category)  
    df = file.drop(columns=['ReceivedTime'])

    # cleaning and resampling
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_standardized.index = df.index

    # resampling and interpolation
    resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample("1min").mean().interpolate(method="linear")
    
    return resample


def raw_input_data_VCD(categories):
    results = Parallel(n_jobs=-1)(delayed(process_category)(category) for category in tqdm(categories))
    combined_df = pd.concat(results, axis=1)
    return combined_df


def create_overlapping_windows(data, window_size, overlap_size):
    start_times = pd.date_range(start=data.index[0], end=data.index[-1], freq=window_size - overlap_size)
    end_times = start_times + window_size
    windows = [data.loc[start:end] for start, end in zip(start_times, end_times)]
    return windows
 

def fit_arima_and_get_sigma2(column_data, order=(1, 1, 1)):
    if column_data.nunique() <= 5:  # Skip constant or near-constant series
        return np.nan
    try:
        model = ARIMA(column_data, order=order)  
        fitted_model = model.fit(start_params=np.zeros(3)) 
        return fitted_model.params.iloc[-1]
    except:
        return np.nan  


def extract_sigma2(windows):
    last_times = [df.index[-1] for df in windows]
    variables = windows[0].columns

    results = Parallel(n_jobs=-1)(
        delayed(lambda df: [fit_arima_and_get_sigma2(df[col]) for col in df.columns])(df)
        for df in tqdm(windows, desc="Processing Fit ARIMA on Time Windows")
    )

    sigma2_df = pd.DataFrame(results, index=last_times, columns=variables)
    return sigma2_df


def calculate_accuracy(mtad_df, faults, k=3):
    # step01: flag the anomalies
    mu = mtad_df['Distance'].mean()
    std = mtad_df['Distance'].std()
    mtad_df['Anomaly'] = mtad_df["Distance"] > (mu + k * std)
    
    # Steo02: locate the fault time
    mtad_df['Fault'] = False
    mtad_df["Time"] = pd.to_datetime(mtad_df["Time"])
    for f in faults:
        # find the upper closest time to the fault time
        upper_time = mtad_df[mtad_df['Time'] > f]['Time'].min()
        mtad_df.loc[mtad_df['Time'] == upper_time, 'Fault'] = True
    
    # Step03: calculate the true positive, false positive, true negative, false negative (return the matrix
    TP = mtad_df[(mtad_df['Anomaly'] == True) & (mtad_df['Fault'] == True)].shape[0]
    FP = mtad_df[(mtad_df['Anomaly'] == True) & (mtad_df['Fault'] == False)].shape[0]
    TN = mtad_df[(mtad_df['Anomaly'] == False) & (mtad_df['Fault'] == False)].shape[0]
    FN = mtad_df[(mtad_df['Anomaly'] == False) & (mtad_df['Fault'] == True)].shape[0]
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # return the matrix
    output = {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "Accuracy": round(accuracy, 2), "Precision": round(precision, 2)}
    
    return output


def calculate_confusion_matrix(mtad_df, faults, k=3):
    # step01: flag the anomalies
    mu = mtad_df['Distance'].mean()
    std = mtad_df['Distance'].std()
    mtad_df['Anomaly'] = mtad_df["Distance"] > (mu + k * std)
    
    # Steo02: locate the fault time
    mtad_df['Fault'] = False
    mtad_df["Time"] = pd.to_datetime(mtad_df["Time"])
    for f in faults:
        # find the upper closest time to the fault time
        upper_time = mtad_df[mtad_df['Time'] > f]['Time'].min()
        mtad_df.loc[mtad_df['Time'] == upper_time, 'Fault'] = True
    
    return confusion_matrix(mtad_df['Fault'], mtad_df['Anomaly']) 


def plot_wasserstein_time_series_flag_anomalies(time_stamps, wasserstein_distances, file_name, k=3, reset=None, title="MTAD Analysis"):
    # Convert timestamps to pandas datetime
    time_series_df = pd.DataFrame({"Time": time_stamps, "WTDA": wasserstein_distances})
    # save the time_series_df to csv file

    # flag the anomalies
    time_series_df['Anomaly'] = time_series_df['WTDA'] > time_series_df['WTDA'].mean() + k*time_series_df['WTDA'].std()
    anomalies = time_series_df[time_series_df['Anomaly']]


    # Plot the time series
    plt.figure(figsize=(18, 6))
    plt.plot(time_series_df['Time'], time_series_df['WTDA'], color="blue", linestyle="-", marker="o", markersize=1 
             #label='WTDA'
             )
    plt.scatter(anomalies['Time'], anomalies['WTDA'], color='red',
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
    
    # Save plot
    plt.savefig(fr"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\VCD\{file_name}.png")
    plt.close()
    
    return time_series_df


raw_data = raw_input_data_VCD(["Analogs"])
windows = create_overlapping_windows(raw_data, window_size=pd.Timedelta(minutes=94), overlap_size=pd.Timedelta(minutes=23))[:-1]
sigma2_result = extract_sigma2(windows)
sigma2_result.to_csv("C:/Users/Sina.Mokhtar.XLSCIENTIFIC/Documents/Problems/XVI/VCD/sigma2_result_analogs.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(sigma2_result.T, cmap="coolwarm", robust=True, 
            xticklabels=50, yticklabels=True)  # Transpose to have time on x-axis

plt.title("Heatmap of Sigma2 Values")
plt.xlabel("Time")
plt.ylabel("Variables")
plt.xticks(rotation=45)
plt.show()

# chage sigma2_resul to a series that contain the sum of each column
sigma2_result_sum = sigma2_result.sum(axis=1)

# Plot the time series
plt.figure(figsize=(6, 3))
plt.plot(sigma2_result_sum.index, sigma2_result_sum, label="Sum of Sigma2", color='b')

plt.title("Time Series of Summed Sigma2 Values")
plt.xlabel("Time")
plt.ylabel("Summed Sigma2")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.show()


sigma2_result_sum.index


plot_wasserstein_time_series_flag_anomalies(time_stamps = sigma2_result_sum.index,
                                             wasserstein_distances = sigma2_result_sum, 
                                             file_name = 'VCD_analog_94min', 
                                             k=3, 
                                             reset = anomaly_times, 
                                             title="VCD Analysis Sum of Sigma2 Values")



sigma2_result_sum_df = sigma2_result_sum.to_frame()
sigma2_result_sum_df.reset_index(inplace=True)
sigma2_result_sum_df.columns = ["Time", "Distance"]
calculate_accuracy(sigma2_result_sum_df, faults=anomaly_times, k=3)
calculate_confusion_matrix(mtad_df=sigma2_result_sum_df, faults=anomaly_times, k=3)

from scipy.stats import norm
import numpy as np
import pandas as pd

def compute_p_values(df):
    p_values = df.copy()
    for col in df.columns:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std > 0:  # Avoid division by zero
            p_values[col] = 1 - norm.cdf(df[col], loc=col_mean, scale=col_std)
        else:
            p_values[col] = np.nan  # Assign NaN where std is zero
    return p_values

# Compute p-values for each value in the DataFrame
sigma2_p_values = 1 - compute_p_values(sigma2_result)

plt.figure(figsize=(8, 6))
sns.heatmap(sigma2_p_values.T, cmap="coolwarm", robust=True, 
            xticklabels=50, yticklabels=True)  # Transpose to have time on x-axis

plt.title("Heatmap of Sigma2 Values")
plt.xlabel("Time")
plt.ylabel("Variables")
plt.xticks(rotation=45)
plt.show()

sigma2_p_values_mean = sigma2_p_values.mean(axis=1)

plot_wasserstein_time_series_flag_anomalies(time_stamps = sigma2_p_values_mean.index,
                                             wasserstein_distances = sigma2_p_values_mean, 
                                             file_name = 'VCD_analog_94min_pvalue', 
                                             k=2, 
                                             reset = anomaly_times, 
                                             title="VCD Analysis")


sigma2_p_values_mean = sigma2_p_values_mean.to_frame()
sigma2_p_values_mean.reset_index(inplace=True)
sigma2_p_values_mean.columns = ["Time", "Distance"]
calculate_accuracy(sigma2_p_values_mean, faults=anomaly_times, k=3)
calculate_confusion_matrix(mtad_df=sigma2_p_values_mean, faults=anomaly_times, k=3)
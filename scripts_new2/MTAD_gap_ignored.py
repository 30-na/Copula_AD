# from EDA.my_functions import *
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

#from EDA.my_functions import *

# reset times and dates

reset = pd.DatetimeIndex([
'2023-08-11 01:30:00',
'2023-08-26 15:38:41',
'2023-09-13 17:40:00',
'2023-09-16 09:26:00',
'2023-10-28 18:40:00', 
'2024-02-25 06:22:04',
'2023-08-04 19:23:00', # operator
'2023-09-27 19:01:00', # operator
'2023-12-04 16:57:00'], # operator
dtype='datetime64[ns, UTC]', freq=None)


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

anomaly_times = pd.DatetimeIndex([
"2023-09-15 21:41:09",
"2023-09-15 21:41:11",
"2023-09-15 21:41:11",
"2023-12-11 08:19:05",
"2023-12-11 14:54:17",
"2023-12-11 14:54:18",
"2023-12-11 14:54:25",
"2023-12-11 14:54:54",
"2023-12-11 19:35:26",
"2023-12-11 21:02:37",
"2023-12-11 23:58:31",
"2023-12-11 23:59:14",
"2023-12-12 00:00:22",
"2024-01-17 20:20:34",
#"2023-07-01 15:10:34",
"2023-08-04 19:23:11",
"2023-08-11 01:54:03",
"2023-08-26 15:38:41",
"2023-09-13 17:40:58",
"2023-09-16 09:26:18",
"2023-09-27 19:01:17",
"2023-10-28 18:37:35",
'2023-12-04 16:57:40',
"2024-01-18 15:11:46",
"2024-02-25 06:19:51",
"2024-02-25 21:14:15",
"2024-02-25 21:15:27",
"2024-03-26 11:11:17",
"2023-09-11 14:42:50",
"2023-09-24 04:22:49",
"2023-10-07 16:20:01",
"2023-10-15 04:19:57",
"2023-10-30 13:06:15",
"2023-11-20 20:35:34",
"2023-11-27 14:07:45",
"2023-12-11 08:19:05",
"2023-12-11 08:25:25",
"2023-12-11 14:54:01",
"2023-12-11 14:54:25",
"2023-12-11 14:54:54",
"2023-12-11 19:35:26",
"2023-12-11 21:02:37",
"2023-12-11 23:58:31",
"2023-12-11 23:59:14",
"2023-12-12 00:00:22",
"2024-01-14 23:57:39",
"2024-02-01 16:35:48",
"2024-02-02 20:26:29",
"2024-02-05 14:56:55",
"2024-02-19 00:43:32",
"2024-03-16 15:42:23",
#"2023-07-09 13:43:37",
"2023-07-10 17:37:45",
"2023-07-20 16:48:10",
"2023-08-24 20:01:05",
"2023-08-31 18:37:07",
"2023-09-06 07:28:31",
"2023-09-07 21:55:36",
"2023-09-15 21:40:55",
"2023-09-16 09:49:05",
"2023-09-26 16:55:41",
"2023-10-08 08:07:30",
"2023-10-20 20:03:39",
"2023-11-23 19:31:03",
"2023-12-12 20:43:40",
"2023-12-12 20:49:54",
"2023-12-12 20:50:52",
"2023-12-12 20:51:16",
"2023-12-12 20:51:34",
"2023-12-13 16:39:28",
"2023-12-19 16:55:58",
"2024-02-25 15:50:44",
"2024-02-29 21:27:57",
"2024-03-05 06:48:56",
"2023-08-30 15:33:30",
"2023-09-08 04:51:42",
"2023-09-25 16:39:33",
"2023-11-12 04:16:37",
"2023-11-12 16:50:24",
"2023-11-30 21:00:29",
"2023-12-01 17:58:44",
"2023-12-01 20:49:59"],
dtype='datetime64[ns, UTC]', freq=None)
# Functions

def read_file_category_XVI(category, time_index='UtcTime'):
    # Read the CSV file and make a new dataframe
    path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\XVIData"
    file_name = category + ".csv"
    file = pd.read_csv(os.path.join(path, file_name))
    file[time_index] = pd.to_datetime(file[time_index], format='ISO8601')
    file = file.set_index(time_index)
    return file


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


def transform_to_long_format_satsam(data):
    """
    Converts a wide-format DataFrame into long format for easier analysis.

    Parameters:
    data (pandas.DataFrame): A DataFrame with 'UtcTime' as the index and multiple prediction columns.

    Returns:
    pandas.DataFrame: A transformed DataFrame with columns ['time', 'model', 'category', 'variable', 'prediction'].

    """
    long_data = (data.reset_index().melt(id_vars='UtcTime', var_name="full_column", value_name="prediction"))
    
    # Rename the index column to 'time'
    long_data.rename(columns={"UtcTime": "time"}, inplace=True)

    # Extract components from the column names
    long_data["model"] = long_data["full_column"].str.extract(r"^(.*?)_")
    long_data["category"] = long_data["full_column"].str.extract(r"_(.*?)%")
    long_data["variable"] = long_data["full_column"].str.extract(r"%(.*)$")

    # Reorder and drop the original column name
    long_data = long_data[["time", "model", "category", "variable", "prediction"]]
    long_data["variable"] = long_data["category"] + "_" + long_data["variable"]
    long_data['time'] = pd.to_datetime(long_data['time'])
    return long_data


def compute_distance_matrix(predictions, distance, threshold=None):
    
    if distance == "Hamming":
        dist_matrix = squareform(pdist(predictions, metric=lambda a, b: np.sum(a != b)))

    elif distance == "Euclidean":
        dist_matrix = squareform(pdist(predictions, metric='euclidean'))

    elif distance == "Jaccard":
        predictions = predictions.astype(bool)
        dist_matrix = squareform(pdist(predictions, metric='jaccard'))
    
    elif distance == "Manhattan":
        dist_matrix = squareform(pdist(predictions, metric='cityblock'))
    
    elif distance == "Cosine":
        dist_matrix = squareform(pdist(predictions, metric='cosine'))
    
    elif distance == "Correlation":
        # if np.any(np.std(predictions, axis=1) == 0):  # Check row-wise variance
        #     dist_matrix = np.ones((len(predictions), len(predictions)))
        # else: 
        #     dist_matrix = squareform(pdist(predictions, metric='correlation'))
        dist_matrix =  1 - abs(np.corrcoef(predictions))
        # num_nodes = len(predictions)
        # dist_matrix = np.ones((num_nodes, num_nodes))  # Initialize with max distance

        # for i in range(num_nodes):
        #     for j in range(i + 1, num_nodes):
        #         if np.std(predictions[i]) == 0 or np.std(predictions[j]) == 0:
        #             corr = 0  # Undefined correlation, treat as 0 (max distance)
        #         else:
        #             corr, _ = pearsonr(predictions[i], predictions[j])
                
        #         dist = 1 - abs(corr)  # Convert correlation to distance
        #         dist_matrix[i, j] = dist
        #         dist_matrix[j, i] = dist  # Maintain symmetry
    
    else:
        raise ValueError("Undefined Distance Metric")
    
    if threshold is not None:
        dist_matrix[dist_matrix > threshold] = 0

    return dist_matrix

 
def process_time_window(group, nodes, distance, threshold):
    
    unique_nodes = group[nodes].unique()
    
    # Convert predictions to a matrix (rows: nodes, cols: features)
    predictions = np.array([group[group[nodes] == node]['prediction'].values for node in unique_nodes])
    
    adjacency_matrix = compute_distance_matrix(predictions, distance, threshold)
    
    return adjacency_matrix


def create_graphs(data, time_interval, nodes, layer, distance, threshold=None, n_jobs=-1):
    layers = data[layer].unique()
    temporal_graphs = {}

    for l in tqdm(layers):
        layer_data = data[data[layer] == l].copy()
        layer_data = layer_data.set_index('time')
        grouped = layer_data.groupby(pd.Grouper(freq=time_interval))

        # Parallel processing of time windows
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_time_window)(group, nodes, distance, threshold)
            for _, group in grouped
        )

        temporal_graphs[l] = {time: adj for (time, adj) in zip(grouped.groups.keys(), results) if adj is not None}

    return temporal_graphs


def process_time_window_wide(prediction_matrix, distance, threshold):
    # prediction_matrix shape: (num_variables, window_size)
    adjacency_matrix = compute_distance_matrix(prediction_matrix, distance, threshold)
    return adjacency_matrix


def create_graphs_by_sample(data, window_size, nodes, layer, distance, threshold=None, n_jobs=-1):
    layers = data[layer].unique()
    temporal_graphs = {}

    for l in tqdm(layers):
        layer_data = data[data[layer] == l].copy()
        layer_data = layer_data.pivot_table(index='time', columns=nodes, values='prediction')
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
            delayed(process_time_window_wide)(window, distance, threshold)
            for window in windows
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
    geodesic_distances (dict): A nested dictionary {node -> {time_window -> distance_matrix}}.

    Returns:
    dict: A nested dictionary {node -> {time_window -> persistence_diagram}}.

    """
    persistence_diagrams = {}

    for node, time_windows in geodesic_distances.items():
        node_diagrams = {}

        for time_window, distance_matrix in time_windows.items():
            # Ensure the distance matrix is a NumPy array
            distance_matrix = np.array(distance_matrix)

            # Compute persistence diagram using Ripser
            result = ripser(distance_matrix, distance_matrix=True)
            diagram = result['dgms']

            node_diagrams[time_window] = diagram

        persistence_diagrams[node] = node_diagrams

    return persistence_diagrams


def stack_persistence_diagrams_by_time(persistence_diagrams):
    """
    Stacks persistence diagrams across time, assigning layer IDs for each model and dimension.

    Parameters:
    persistence_diagrams (dict): A nested dictionary {model -> {time_stamp -> persistence_diagram}}.

    Returns:
    dict: A dictionary {time_stamp -> stacked_persistence_diagram} where each persistence diagram is 
          an Nx3 NumPy array with (layer_id, birth, death).

    Notes:
    - Assigns a unique layer ID for each (model, dimension) pair.
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
    Computes the Wasserstein distance for each layer separately and sums them up.
    
    :param stacked_diagrams: Dictionary {timestamp: persistence diagram}
                             Each persistence diagram is an Nx3 array with [layer, birth, death].
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
        layers = np.unique(pd1[:, 0])

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


def plot_wasserstein_time_series_flag_anomalies(time_stamps, wasserstein_distances, file_name, k=3, reset=None, title="MTAD Analysis"):
    # Convert timestamps to pandas datetime
    time_series_df = pd.DataFrame({"Time": time_stamps, "WTDA": wasserstein_distances})
    # save the time_series_df to csv file

    # flag the anomalies
    time_series_df['Anomaly'] = time_series_df['WTDA'] > time_series_df['WTDA'].mean() + k*time_series_df['WTDA'].std()
    anomalies = time_series_df[time_series_df['Anomaly']]


    # Plot the time series
    plt.figure(figsize=(14, 6))
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
    plt.savefig(fr"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\performance\{file_name}.png")
    plt.close()
    
    return time_series_df


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



def raw_input_data():
    satsam_file = read_and_merge_csv(folder_path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\outputArchive")
    satsam_variables_dict = variable_selection(df=satsam_file)
    combined_df = pd.DataFrame()

    for category, variables in tqdm(satsam_variables_dict.items()):
        file = read_file_category_XVI(category)  
        
        # Select columns that match the variables
        selected_columns = [col for col in file.columns if col in variables]

        # Subset and rename columns
        subset_df = file[selected_columns]
        subset_df.columns = [f"{category}_{col}" for col in selected_columns]


        # cleaning and resampling
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(scaler.fit_transform(subset_df), columns=subset_df.columns)
        df_standardized.index = subset_df.index
    
        # resampling and interpolation
        resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample("1min").mean().interpolate(method="linear")
        print(resample.shape)
        # Combine into final DataFrame
        combined_df = pd.concat([combined_df, resample], axis=1)
        #combined_df = combined_df.dropna()
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
        print(resample.shape)

        combined_df = pd.concat([combined_df, resample], axis=1)

    return combined_df


def raw_input_data_withgap():
    satsam_file = read_and_merge_csv(folder_path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\outputArchive")
    satsam_variables_dict = variable_selection(df=satsam_file)
    combined_df = pd.DataFrame()
    gap_series = None
    sampling_rate_estimated = False

    for category, variables in tqdm(satsam_variables_dict.items()):
        file = read_file_category_XVI(category)  
        selected_columns = [col for col in file.columns if col in variables]
        subset_df = file[selected_columns]
        subset_df.columns = [f"{category}_{col}" for col in selected_columns]

        # Ensure datetime index and sort
        subset_df = subset_df[~subset_df.index.duplicated(keep='first')]
        subset_df = subset_df.sort_index()

        # Estimate expected points per minute only once
        if not sampling_rate_estimated:
            deltas = subset_df.index.to_series().diff().dropna().dt.total_seconds()
            mode_interval = deltas.mode().iloc[0]
            expected_per_minute = int(60 / mode_interval)
            sampling_rate_estimated = True

            # Calculate gap once using full index
            count_per_minute = subset_df.resample("1min").size()
            gap_series = expected_per_minute - count_per_minute
            gap_series[gap_series < 0] = 0
            gap_series.name = "gap"

        # Standardize and resample
        scaler = StandardScaler()
        df_standardized = pd.DataFrame(scaler.fit_transform(subset_df), columns=subset_df.columns, index=subset_df.index)
        resample = df_standardized.resample("1min").mean().interpolate(method="linear")

        # Combine category data
        combined_df = pd.concat([combined_df, resample], axis=1)

    # Add gap column after combining all
    combined_df["gap"] = gap_series
    return combined_df


def transform_to_long_format_rawdata(data):
    long_data = (data.reset_index().melt(id_vars='UtcTime', var_name="variable", value_name="prediction"))
    long_data.rename(columns={"UtcTime": "time"}, inplace=True)
    # add a column name model and assigne "rawdata"
    long_data["model"] = "rawdata"
    long_data['time'] = pd.to_datetime(long_data['time'])
    return long_data


def save_correlation_matrices(temporal_graphs, save_path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\performance\corr_matrices"):
    os.makedirs(save_path, exist_ok=True)

    for layer, graphs in temporal_graphs.items():
        for date, corr_matrix in graphs.items():
            
            mask = np.isnan(corr_matrix)
            cmap = plt.get_cmap("coolwarm")
            cmap.set_bad(color='gray')  # Set NaN values to gray

            plt.figure(figsize=(6, 6))  # Set rectangular aspect ratio
            plt.imshow(np.ma.masked_where(mask, corr_matrix), cmap=cmap, interpolation='nearest', aspect='auto')
            plt.colorbar(label='Correlation Coefficient')
            plt.title(f'Correlation Matrix - {date}')
            plt.xlabel('Nodes')
            plt.ylabel('Nodes')

            filename = os.path.join(save_path, f"{date.strftime('%Y-%m-%d_%H-%M-%S')}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
                


def mtad_analysis(df, threshold, file_name, title, time_interval='94min', nodes='variable', layer='model', distance='Correlation', reset=None, k=2, n_jobs=-1):
    
    # 1: Create temporal networks
    
    graphs = create_graphs(data=df, time_interval=time_interval, nodes=nodes, layer=layer, distance=distance, threshold=threshold, n_jobs=n_jobs)
    #graphs = create_graphs(data=long_df_raw, time_interval=time_interval, nodes=nodes, layer=layer, distance=distance, threshold=threshold, n_jobs=n_jobs)
    #list(graphs["rawdata"].keys())[1:100]
  
    # save_correlation_matrices(graphs)


    # 2: Compute geodesic distances
    geodesic_distances = compute_geodesic_distances_scipy(graphs)
    
    # 3: Generate the Persistence Diagram
    persistence_diagrams = compute_persistence_diagrams(geodesic_distances)
    
    # 4: Stack Persistence Diagrams
    stacked_diagrams = stack_persistence_diagrams_by_time(persistence_diagrams)
    
    # 5: Compute the Wasserstein Distance
    time_stamps, distances = compute_wasserstein_distances(stacked_diagrams)
    
    # 6: Plot the time series and save CSV
    distances_df = pd.DataFrame({'Time': time_stamps, 'Distance': distances})

    #distances_df.to_csv(fr"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\performance\{file_name}.csv", index=False)
    
    plot_wasserstein_time_series_flag_anomalies(time_stamps, distances, file_name=file_name, reset=reset, k=k, title=title)
    
    return distances_df



def mtad_analysis_by_sample(df, threshold, file_name, title, window_size, nodes='variable', layer='model', distance='Correlation', reset=None, k=2, n_jobs=-1):
    
    # 1: Create temporal networks
    graphs = create_graphs_by_sample(
        data=df,
        window_size=window_size,
        nodes=nodes, 
        layer=layer,
        distance=distance,
        threshold=threshold,
        n_jobs=n_jobs
    )

    type(graphs)
    graphs.items()
    # 2: Compute geodesic distances
    geodesic_distances = compute_geodesic_distances_scipy(graphs)
    
    # 3: Generate the Persistence Diagram
    persistence_diagrams = compute_persistence_diagrams(geodesic_distances)
    
    # 4: Stack Persistence Diagrams
    stacked_diagrams = stack_persistence_diagrams_by_time(persistence_diagrams)
    
    # 5: Compute the Wasserstein Distance
    time_stamps, distances = compute_wasserstein_distances(stacked_diagrams)
    
    # 6: Plot the time series and save CSV
    distances_df = pd.DataFrame({'Time': time_stamps, 'Distance': distances})

    #distances_df.to_csv(fr"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\performance\{file_name}.csv", index=False)
    
    plot_wasserstein_time_series_flag_anomalies(time_stamps, distances, file_name=file_name, reset=reset, k=k, title=title)
    
    return distances_df


def get_anomaly_labels(mtad_df, faults, k):
    df = pd.DataFrame()
    df['Time'] = pd.to_datetime(mtad_df['Time'])
    df['Distance'] = mtad_df['Distance']
    df.set_index('Time', inplace=True)

    mu = df['Distance'].mean()
    std = df['Distance'].std()
    df['y_pred'] = (df['Distance'] > (mu + k * std)).astype(int)

    df['y_true'] = 0
    for f in faults:
        upper_time = df.index[df.index > f].min()
        if pd.notna(upper_time):
            df.at[upper_time, 'y_true'] = 1

    return df[['y_true', 'y_pred']]


def tune_k_auc(mtad_df, faults, k_range=np.arange(1.0, 5.1, 0.1), plot=False, file_name="all_roc_curves.png"):
    auc_scores = []
    best_auc = -1
    best_k = None
    all_curves = []

    for k in k_range:
        labels_df = get_anomaly_labels(mtad_df, faults, k)
        y_true = labels_df['y_true'].values
        y_pred = labels_df['y_pred'].values

        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        all_curves.append((k, fpr, tpr, auc))
        auc_scores.append((k, auc))

        if auc > best_auc:
            best_auc = auc
            best_k = k

    if plot and all_curves:
        base_path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\performance"
        file_path = os.path.join(base_path, file_name)

        # Prepare data for second plot
        ks = [k for k, _ in auc_scores]
        aucs = [auc for _, auc in auc_scores]

        # Plot side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        best_k, best_fpr, best_tpr, best_auc_val = max(all_curves, key=lambda x: x[3])
        middle_fprs = []
        middle_tprs = []
        for _, fpr, tpr, _ in all_curves:
            middle_fprs.append(fpr[1])
            middle_tprs.append(tpr[1])
        fpr_sorted, tpr_sorted = zip(*sorted(zip(middle_fprs, middle_tprs)))
        fpr_sorted = [0.0] + list(fpr_sorted) + [1.0]
        tpr_sorted = [0.0] + list(tpr_sorted) + [1.0]

        
        ax1.plot(best_fpr[1], best_tpr[1], 'ro', markersize=8, alpha=.7)
        ax1.plot(fpr_sorted, tpr_sorted, marker='o')
        #ax1.plot(fpr_curve, tpr_curve, color=color, alpha=.8)
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.set_aspect('equal')
        ax1.grid(True)

        annotation_text = f'k = {best_k:.2f}\nAUC = {best_auc:.4f}'
        ax1.text(0.65, 0.05, annotation_text, fontsize=10, color='black',
                 bbox=dict(facecolor='white', edgecolor='black'))

        # AUC vs k (right)
        ax2.plot(ks, aucs, marker='o', label='AUC')
        ax2.scatter(best_k, best_auc, color='red', s=80, label='Best AUC')
        #ax2.text(best_k, best_auc + 0.01, f'best k={best_k:.1f}', ha='center', fontsize=10, color='black')
        ax2.set_xlabel('k (Standard Deviation Threshold)')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('AUC vs k')
        ax2.grid(True)
        #ax2.set_aspect('equal')

        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

    return {
        "BestK": best_k,
        "BestAUC": best_auc,
        "AUC_Scores": auc_scores
    }


def mtad_performance(directory, faults, k=3):

    results = []

    for filename in os.listdir(directory):
        
        if filename.startswith("MTAD_satsam") and filename.endswith(".csv"):
            
            file_path = os.path.join(directory, filename)
            # Extract metadata from filename
            parts = filename.split("_")
            #output = parts[0]
            #source = part[1]  # First word before "_"
            distance = parts[2] 
            model = parts[3].split(".")[0] 
    
            # Read the CSV file
            mtad_df = pd.read_csv(file_path)
            accuracy_results = calculate_accuracy(mtad_df, faults, k)
            
            results.append([distance, model] + list(accuracy_results.values()))
            
    
    # Create DataFrame
    columns = ["Distance", "Model", "TP", "FP", "TN", "FN", "Accuracy", "Precision"]
    df = pd.DataFrame(results, columns=columns)
    print(df.to_latex(index=False))
    return df


def flag_anomlies(time_series, time_stamps, k=3):
    anomlaies = time_series > time_series.mean() + k*time_series.std()
    return anomlaies


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
    

def plot_k_vs_auc(auc_result):
    auc_scores = auc_result["AUC_Scores"]
    best_k = auc_result["BestK"]
    best_auc = auc_result["BestAUC"]

    ks = [k for k, _ in auc_scores]
    aucs = [auc for _, auc in auc_scores]

    plt.figure(figsize=(9, 5))
    plt.plot(ks, aucs, marker='o', label='AUC')

    # Highlight the best point
    plt.scatter(best_k, best_auc, color='red', s=80, label='Best AUC')
    plt.text(best_k, best_auc + 0.01, f'best k={best_k:.1f}', ha='center', fontsize=10, color='black')

    plt.xlabel('k (Standard Deviation Threshold)')
    plt.ylabel('AUC Score')
    plt.title('AUC vs k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_accuracy_next_window(mtad_df, faults, k=3):
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


def get_interpolation_mask(df, freq="1min"):
    df = df[~df.index.duplicated(keep='first')]
    df_resampled = df.resample(freq).mean()
    interpolated_mask = df_resampled.isna().any(axis=1).astype(int)
    return interpolated_mask


#################

if __name__ == "__main__":
    df_raw = raw_input_data_ignore_gap()
    df_raw = df_raw.dropna()
    df_raw = df_raw.loc[df_raw.index < "2024-03-27"]

    long_df_raw = transform_to_long_format_rawdata(df_raw)

    df_raw_30sample_Correlation = mtad_analysis_by_sample(
        df=long_df_raw, 
        window_size=30, 
        distance='Correlation', 
        threshold=None, 
        file_name='MTAD_rawdata_Correlation_30_ignore_gap', 
        reset=anomaly_times, 
        k=2, 
        title='TAD on Raw Data from XVI using Correlation Distance (30 sample)',
        n_jobs=-1)

    df_raw_30sample_Euclidean = mtad_analysis_by_sample(
            df=long_df_raw, 
            window_size=30, 
            distance='Euclidean', 
            threshold=None, 
            file_name='MTAD_rawdata_Euclidean_30_ignore_gap', 
            reset=anomaly_times, 
            k=2, 
            title='TAD on Raw Data from XVI using Euclidean Distance (30 sample)',
            n_jobs=-1)
    
    df_raw_30sample_Manhattan = mtad_analysis_by_sample(
            df=long_df_raw, 
            window_size=30, 
            distance='Manhattan', 
            threshold=None, 
            file_name='MTAD_rawdata_Manhattan_30_ignore_gap', 
            reset=anomaly_times, 
            k=2, 
            title='TAD on Raw Data from XVI using Manhattan Distance (30 sample)',
            n_jobs=-1)
    

    df_raw_30sample_Cosine = mtad_analysis_by_sample(
            df=long_df_raw, 
            window_size=30, 
            distance='Cosine', 
            threshold=None, 
            file_name='MTAD_rawdata_Cosine_30_ignore_gap', 
            reset=anomaly_times, 
            k=2, 
            title='TAD on Raw Data from XVI using Cosine Distance (30 sample)',
            n_jobs=-1)

    get_interpolation_mask(df_raw, freq="1min")

    def plot_distance_with_mask(df_distance, mask_series, t, mask_color='lightgray'):
        fig, ax = plt.subplots(figsize=(15, 5))

        # Draw gray vertical lines first, with lower zorder
        for ts, val in mask_series.items():
            if val == 1:
                ax.axvline(ts, color=mask_color, alpha=0.4, zorder=1, linewidth=0.1)

        # Draw scatter points on top, with higher zorder
        ax.scatter(df_distance['Time'], df_distance['Distance'], label='Distance', s=2, color='blue', zorder=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Distance")
        ax.set_title(t)
        ax.legend()
        plt.tight_layout()
        plt.show()


    
    plot_distance_with_mask(df_raw_30sample_Manhattan, 
                            get_interpolation_mask(df_raw),
                            t="Manhattan Distance with Interpolated Time Mask Overlay (30 Samles Windows)")
    plot_distance_with_mask(df_raw_30sample_Cosine, 
                        get_interpolation_mask(df_raw),
                        t="Cosine Distance with Interpolated Time Mask Overlay (30 Samles Windows)")


    df_raw_1440sample_Correlation = mtad_analysis_by_sample(
        df=long_df_raw, 
        window_size=1440, 
        distance='Correlation', 
        threshold=None, 
        file_name='MTAD_rawdata_Correlation_1440_ignore_gap', 
        reset=anomaly_times, 
        k=2, 
        title='TAD on Raw Data from XVI using Correlation Distance (1440 sample)',
        n_jobs=-1)

    df_raw_1440sample_Correlation = mtad_analysis_by_sample(
            df=long_df_raw, 
            window_size=1440, 
            distance='Euclidean', 
            threshold=None, 
            file_name='MTAD_rawdata_Euclidean_1440_ignore_gap', 
            reset=anomaly_times, 
            k=2, 
            title='TAD on Raw Data from XVI using Euclidean Distance (1440 sample)',
            n_jobs=-1)

    ########### RAW DATA with Interpolation
    df_raw_interpolated = raw_input_data()
    df_raw_interpolated = df_raw_interpolated.dropna()
    df_raw_interpolated = df_raw_interpolated.loc[df_raw_interpolated.index < "2024-03-27"]

    long_df_raw_interpolated = transform_to_long_format_rawdata(df_raw_interpolated)

    df_raw_30min_Correlation = mtad_analysis(df=long_df_raw_interpolated, 
                time_interval='30min', 
                nodes='variable', 
                layer='model', 
                distance='Correlation', 
                threshold=None, 
                file_name='MTAD_rawdata_Correlation_30m', 
                reset=anomaly_times, 
                k=2, 
                title=f'TAD on Raw Data from XVI using Correlation Distance (30 min interval)',
                n_jobs=-1)
    

    df_raw_30min_Euclidean = mtad_analysis(df=long_df_raw_interpolated, 
                time_interval='30min', 
                nodes='variable', 
                layer='model', 
                distance='Euclidean', 
                threshold=None, 
                file_name='MTAD_rawdata_euclidean_30m', 
                reset=anomaly_times, 
                k=2, 
                title=f'TAD on Raw Data from XVI using Euclidean Distance (30 min interval)',
                n_jobs=-1)
    

    
    df_raw_30min_Manhattan = mtad_analysis(df=long_df_raw_interpolated, 
            time_interval='30min', 
            nodes='variable', 
            layer='model', 
            distance='Manhattan', 
            threshold=None, 
            file_name='MTAD_rawdata_Manhattan_30m', 
            reset=anomaly_times, 
            k=2, 
            title=f'TAD on Raw Data from XVI using Manhattan Distance (30 min interval)',
            n_jobs=-1)
    
    plot_distance_with_mask(df_raw_30min_Manhattan, 
                            get_interpolation_mask(df_raw),
                            t="Manhattan Distance with Interpolated Time Mask Overlay (30 Minutes Windows)")
   
    df_raw_30min_Cosine = mtad_analysis(df=long_df_raw_interpolated, 
            time_interval='30min', 
            nodes='variable', 
            layer='model', 
            distance='Cosine', 
            threshold=None, 
            file_name='MTAD_rawdata_Cosine_30m', 
            reset=anomaly_times, 
            k=2, 
            title=f'TAD on Raw Data from XVI using Cosine Distance (30 min interval)',
            n_jobs=-1)

    
    plot_distance_with_mask(df_raw_30min_Euclidean, get_interpolation_mask(df_raw))
    
    df_raw_1440min_Correlation = mtad_analysis(df=long_df_raw_interpolated, 
            time_interval='1440min', 
            nodes='variable', 
            layer='model', 
            distance='Correlation', 
            threshold=None, 
            file_name='MTAD_rawdata_Correlation_1440m', 
            reset=anomaly_times, 
            k=2, 
            title=f'TAD on Raw Data from XVI using Correlation Distance (1440 min interval)',
            n_jobs=-1)


    df_raw_1440min_Euclidean = mtad_analysis(df=long_df_raw_interpolated, 
                time_interval='1440min', 
                nodes='variable', 
                layer='model', 
                distance='Euclidean', 
                threshold=None, 
                file_name='MTAD_rawdata_euclidean_1440m', 
                reset=anomaly_times, 
                k=2, 
                title=f'TAD on Raw Data from XVI using Euclidean Distance (1440 min interval)',
                n_jobs=-1)


    calculate_accuracy_next_window(df_raw_1800sample_Correlation, faults=anomaly_times, k=2)
    calculate_confusion_matrix(mtad_df=df_raw_1800sample_Correlation, faults=anomaly_times, k=2)

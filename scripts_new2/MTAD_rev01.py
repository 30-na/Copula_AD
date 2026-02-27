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
import h5py


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
        #resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample("1s").mean().interpolate(method="linear").ffill()
        resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample("30s").mean().ffill()
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



def read_faulty_rw_run(path):
    with h5py.File(path, "r") as f:
        t = f["time_s"][:]
        sensors_grp = f["sensors"]
        data = {}
        for name in sorted(sensors_grp.keys()):
            arr = sensors_grp[name][:]
            for i in range(arr.shape[1]):
                data[f"{name}_{i+1}"] = arr[:, i]
        lbl = f["labels"][:]

    if lbl.ndim == 2:
        fault = (lbl.astype(bool).any(axis=1)).astype(int)
    else:
        fault = lbl.astype(int)

    # DataFrame with sensors only
    df_sensors = pd.DataFrame(data)
    df_sensors.insert(0, "time_s", t)

    # DataFrame with time + fault
    df_fault = pd.DataFrame({"time_s": t, "fault": fault})

    return df_sensors, df_fault


def plot_sensors_with_fault_spans_from_split(df_sensors, df_fault, out_dir):
    t = df_sensors["time_s"].to_numpy()
    fault = pd.merge(df_sensors[["time_s"]], df_fault, on="time_s", how="left")["fault"].fillna(0).astype(int).to_numpy()

    starts = np.where((fault[1:] == 1) & (fault[:-1] == 0))[0] + 1
    if fault[0] == 1:
        starts = np.r_[0, starts]
    ends = np.where((fault[1:] == 0) & (fault[:-1] == 1))[0] + 1
    if fault[-1] == 1:
        ends = np.r_[ends, len(fault)]

    bases = sorted({c.rsplit("_", 1)[0] for c in df_sensors.columns if c != "time_s" and c.endswith(("_1","_2","_3"))})

    for idx, base in enumerate(bases, 1):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for k in range(3):
            y = df_sensors[f"{base}_{k+1}"].to_numpy()
            ax = axes[k]
            # ax.plot(t, y, linewidth=1)
            ax.scatter(t, y, s=1)
            for s, e in zip(starts, ends):
                ax.axvspan(t[s], t[e-1], alpha=0.15, color="orange")
            ax.set_ylabel(f"{base}_{k+1}")
        axes[-1].set_xlabel("time_s")
        fig.suptitle(base, y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{idx:02d}_{base}.png"), dpi=150)
        plt.close(fig)


def plot_anomaly_distance(mtad_df, df_fault, out_path, title="Anomaly Distance", ymax=None, threshold=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if {"time_s","fault"}.issubset(df_fault.columns):
        t_fault = df_fault["time_s"].to_numpy()
        fault = df_fault["fault"].to_numpy().astype(int)
    else:
        raise ValueError("df_fault must have columns: time_s, fault")

    starts = np.where((fault[1:] == 1) & (fault[:-1] == 0))[0] + 1
    if fault[0] == 1: starts = np.r_[0, starts]
    ends = np.where((fault[1:] == 0) & (fault[:-1] == 1))[0] + 1
    if fault[-1] == 1: ends = np.r_[ends, len(fault)]

    if "time" in mtad_df.columns:
        x = pd.to_numeric(mtad_df["time"], errors="coerce").to_numpy()
        y = mtad_df.iloc[:, mtad_df.columns.get_loc("time")+1].to_numpy()
    elif "time_s" in mtad_df.columns:
        x = pd.to_numeric(mtad_df["time_s"], errors="coerce").to_numpy()
        y = mtad_df.drop(columns=["time_s"]).iloc[:,0].to_numpy()
    else:
        x = pd.to_numeric(mtad_df.index, errors="coerce").to_numpy()
        y = mtad_df.iloc[:,0].to_numpy()

    order = np.argsort(x)
    x, y = x[order], y[order]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, linewidth=1)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1)
    for s, e in zip(starts, ends):
        ax.axvspan(t_fault[s], t_fault[e-1], alpha=0.15, color="orange")
    ax.set_xlabel("Time")
    ax.set_ylabel(mtad_df.columns[0] if mtad_df.columns.size==1 else "Distance")
    ax.set_title(title)
    if ymax:
        ax.set_ylim(-5, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_mtad_pipeline_CUDATA(file_basename, window_size, k, b, f, a=0.5, d=delta_flat, g=lambda x: 1 / x, ymax=None):
    base_dir = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI"
    file_path = os.path.join(base_dir, "CUData", f"{file_basename}.h5")

    # Load data
    df_sensors, df_fault = read_faulty_rw_run(file_path)
    #df_sensors.columns
    #df_sensors = df_sensors.iloc[:, np.r_[0:10]]

    # Save sensor plot with fault spans
    # plot_sensors_with_fault_spans_from_split(
    #     df_sensors,
    #     df_fault,
    #     out_dir=base_dir
    # )

    # Prepare long-format sensor DataFrame
    long_sensors_df = pd.DataFrame(
        df_sensors.melt(id_vars="time_s", var_name="variable", value_name="value")
        .rename(columns={"time_s": "time"})
    )
    long_sensors_df["layer"] = "rawdata"

    # Run MTAD analysis
    mtad_eucl = mtad_analysis_by_sample(
        df_long=long_sensors_df,
        window_size=window_size,
        distance="Euclidean"
    )

    # Save anomaly distance plot
    

    cu_anomaly_flagged, t = CU_flag_anomalies(df=mtad_eucl, k=k, back=b, forward=f)
    #cu_anomaly_flagged = CU_flag_anomalies_rolling(mtad_eucl, k=k, back=b, forward=f, window=10)
    #cu_anomaly_flagged = CU_flag_anomalies_changepoint(mtad_eucl, k=None, back=0, forward=0, pen=k)
    #cu_anomaly_flagged = CU_flag_anomalies_diff(mtad_eucl, t=k, back=0, forward=0)
    #cu_anomaly_flagged = CU_flag_anomalies_kmeans(mtad_eucl, t=k, back=0, forward=0, n_clusters=2)
    #plt.plot(cu_anomaly_flagged)
    #plt.show()

    # Save anomaly distance plot
    plot_path = os.path.join(base_dir, f"mtad_distance_{file_basename}.png")
    plot_anomaly_distance(
        mtad_eucl,
        df_fault,
        plot_path,
        title=f"TAD on Simulated Data {file_basename[2:]}",
        ymax = ymax,
        threshold = None #t
    )

    cu_anomaly_fault = CU_windowed_fault_change(cu_anomaly_flagged, df_fault, w=window_size)
    result = CU_evaluate_range_based_metrics(cu_anomaly_fault,  alpha=a, delta_fn=d, gamma_fn=g)
    return result


def CU_flag_anomalies(df, k=1, back=0, forward=0):
    mean = df["Distance"].mean()
    std = df["Distance"].std()
    threshold = mean + 2 * k * std

    base_flag = (df["Distance"] > threshold).astype(int).values
    flag = np.zeros(len(df), dtype=int)

    anomaly_indices = np.where(base_flag == 1)[0]

    for i in anomaly_indices:
        start = max(0, i - back)
        end = min(len(df), i + forward + 1)
        flag[start:end] = 1

    return pd.DataFrame({"flag_anomaly": flag}, index=df.index), threshold


def CU_windowed_fault_change(cu_anomaly_flagged, df_fault, w):
    fault = df_fault["fault"].values
    times = df_fault["time_s"].values

    # Step 1: Windowed fault flagging
    window_flags = []
    window_indices = []

    for i in range(w, len(fault), w):
        window = fault[i:i + w]
        if len(window) == 0:
            continue
        flag = int(np.any(window == 1))
        idx = times[i]  # Index of the last point in the window
        window_flags.append(flag)
        window_indices.append(idx)

    windowed_fault = pd.Series(window_flags, index=window_indices)

    # Step 2: Mode change detection
    mode_change = [0]
    for i in range(1, len(windowed_fault)):
        if windowed_fault.iloc[i] != windowed_fault.iloc[i - 1]:
        #if windowed_fault.iloc[i - 1] == 0 and windowed_fault.iloc[i] == 1:

            mode_change.append(1)
        else:
            mode_change.append(0)

    mode_change = pd.Series(mode_change, index=windowed_fault.index)


    # Final dataframe
    result = pd.DataFrame({
        "flag_anomaly": cu_anomaly_flagged.squeeze(),
        "mode_change": mode_change
    }, index=windowed_fault.index)

    return result


def CU_evaluate_range_based_metrics(cu_anomaly_fault,  alpha=0.5, delta_fn=delta_front, gamma_fn=lambda x: 1 / x):
    results = []
    real_ranges = extract_anomaly_ranges(cu_anomaly_fault, 'mode_change')
    predicted_ranges = extract_anomaly_ranges(cu_anomaly_fault, "flag_anomaly")

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
    
    
    results.append({
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    return pd.DataFrame(results)





base_dir = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI"
data_dir = os.path.join(base_dir, "CUData")

w=4
ymax=800
results = []

for file in os.listdir(data_dir):
    if file.endswith(".h5"):
        file_basename = file[:-3]  # remove .h5
        res = run_mtad_pipeline_CUDATA(file_basename, window_size=w, k=1.6, b=0, f=0, d=delta_front, ymax=ymax)
        res["file"] = file_basename
        results.append(res)

df_results = pd.concat(results, ignore_index=True)
#print(df_results)

avg_row = pd.DataFrame({
    "Precision": [df_results["Precision"].mean()],
    "Recall": [df_results["Recall"].mean()],
    "F1": [df_results["F1"].mean()],
    "file": ["AVERAGE"]
})

df_results = pd.concat([df_results, avg_row], ignore_index=True)
print(df_results)



def sweep_k_values(k_values, w=4, ymax=800):
    results_all = []

    for k in k_values:
        results = []
        for file in os.listdir(data_dir):
            if file.endswith(".h5"):
                file_basename = file[:-3]
                res = run_mtad_pipeline_CUDATA(
                    file_basename,
                    window_size=w,
                    k=k,
                    b=0,
                    f=0,
                    d=delta_front,
                    ymax=ymax
                )
                res["file"] = file_basename
                results.append(res)

        df_results = pd.concat(results, ignore_index=True)

        avg_row = {
            "k": k,
            "Precision": df_results["Precision"].mean(),
            "Recall": df_results["Recall"].mean(),
            "F1": df_results["F1"].mean()
        }
        results_all.append(avg_row)

    return pd.DataFrame(results_all)

k_range = np.arange(0.5, 2.3, 0.1)
df_k_results = sweep_k_values(k_range, w=4, ymax=800)
print(df_k_results)


run_mtad_pipeline_CUDATA(file_basename="1_faulty_rw1_0.01torque", window_size=w, k=1.6, b=0, f=0, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="2_faulty_rw2_0.01torque", window_size=w, k=1.2, b=0, f=0, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="2_faulty_rw2_0.01torque", window_size=w, k=1.2, b=0, f=0, ymax=ymax)


run_mtad_pipeline_CUDATA(file_basename="2_faulty_rw2_0.01torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="3_faulty_rw3_0.01torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="4_faulty_rw1_zero_torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="5_faulty_rw2_zero_torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="6_faulty_rw3_zero_torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="7_faulty_rw1_rw2_no_torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="8_faulty_rw2_rw3_no_torque", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="9_faulty_power1", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="10_faulty_power2", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="11_faulty_power3", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="12_faulty_power4", window_size=w, ymax=ymax)
run_mtad_pipeline_CUDATA(file_basename="healthy_rw_run", window_size=w, ymax=ymax)




# usage
path = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\CUData\2_faulty_rw2_0.01torque.h5"
df_sensors, df_fault = read_faulty_rw_run(path)
plot_sensors_with_fault_spans_from_split(df_sensors,df_fault, out_dir=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI")
long_sensors_df = pd.DataFrame(df_sensors.melt(id_vars = "time_s", var_name="variable", value_name="value").rename(columns={"time_s": "time"}))
long_sensors_df["layer"] = "rawdata"

mtad_eucl = mtad_analysis_by_sample(df_long=long_sensors_df, window_size=4, distance="Euclidean")
plot_anomaly_distance(mtad_eucl, df_fault, r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\mtad_distance_2_faulty_rw2_0.01torque.png", title="TAD on Simulated Data (faulty_rw2_0.01torque)")


df_sensors_slice = df_sensors.iloc[1000:1200,:]

for col in df_sensors_slice.columns:
    if col != "time_s":
        plt.figure()
        plt.scatter(df_sensors_slice["time_s"], df_sensors_slice[col], s=5)
        plt.xlabel("time_s")
        plt.ylabel(col)
        plt.title(f"time_s vs {col}")
        plt.show()

plt.figure()
plt.scatter(df_sensors.iloc[1000:1500,:]["time_s"], df_sensors.iloc[1000:1500,:]["rw_motor_torque_Nm_3"].diff(), s=5)
plt.show()


df_sensors_slice = df_sensors.iloc[100:1500,:]
plot_sensors_with_fault_spans_from_split(df_sensors_slice,df_fault, out_dir=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI")

long_sensors_df = pd.DataFrame(df_sensors.melt(id_vars = "time_s", var_name="variable", value_name="value").rename(columns={"time_s": "time"}))
long_sensors_df["layer"] = "rawdata"

mtad_eucl = mtad_analysis_by_sample(df_long=long_sensors_df, window_size=30, distance="Euclidean")
plot_anomaly_distance(mtad_eucl, df_fault, r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\mtad_distance_normal.png", title="TAD on Simulated Data")


#####################################################

df_raw_ignore = raw_input_data_ignore_gap()
df_raw_ignore = df_raw_ignore.dropna()
long_df_raw_ignore = transform_to_long_format_rawdata(df_raw_ignore)


df_raw_fill = raw_input_data_fill_gap()
df_raw_fill = df_raw_fill.dropna()
df_filtered = df_raw_fill[df_raw_fill.index >= '2023-07-10 17:00:00+00:00'] # to be match with SatSAM


long_df_raw_fill = transform_to_long_format_rawdata(df_filtered)


 ## Test on mice data
mtad_eucl_fill_30_mice = mtad_analysis_by_sample(df_long=df_slice, window_size=30, distance="Euclidean")


mtad_corr_fill_30 = mtad_analysis_by_sample(df_long=long_df_raw_fill, window_size=30, distance="Correlation")
mtad_corr_fill_30.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_corr_fill_30.csv")

mtad_eucl_fill_30 = mtad_analysis_by_sample(df_long=long_df_raw_fill, window_size=30, distance="Euclidean")


mtad_eucl_fill_30.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_eucl_fill_30.csv")

mtad_corr_ignore_30 = mtad_analysis_by_sample(df_long=long_df_raw_ignore, window_size=30, distance="Correlation")
mtad_corr_ignore_30.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_corr_ignore_30.csv")

mtad_eucl_ignore_30 = mtad_analysis_by_sample(df_long=long_df_raw_ignore, window_size=30, distance="Euclidean")
mtad_eucl_ignore_30.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_eucl_ignore_30.csv")


mtad_eucl_fill_30 = mtad_analysis_by_sample(df_long=long_df_raw_fill, window_size=30, distance="Correlation", graph_threshold=None)
result = evaluate_k_std_detector(mtad_eucl_ignore_30, k=1.5, fault_times=fault_times, start='2023-08-01', end='2024-04-01')
print(result)


for k in np.arange(0.1, 5, 0.1):
    result = evaluate_k_std_detector(mtad_eucl_fill_30, k=k, fault_times=fault_times, start='2023-08-01', end='2024-04-01')
    print(k)
    print(result)

results = []

for thresh in np.arange(0.1, 1.01, 0.1):
    mtad_result = mtad_analysis_by_sample(
        df_long=long_df_raw_fill,
        window_size=30,
        distance="Correlation",
        graph_threshold=thresh
    )
    eval_df = evaluate_k_std_detector(
        mtad_result,
        k=3,
        fault_times=fault_times,
        start='2023-08-01',
        end='2024-04-01'
    )
    eval_df["threshold"] = round(thresh, 1)
    results.append(eval_df)

threshold_df = pd.concat(results, ignore_index=True)




## Select Variables
vars = ["AttDet_ConvertedBodyRate3", 
        "Momentum_ConvertedBusMomentumBody3", 
        "Refs_ConvertedVelocityWrtEci1",
          "Radio_SdrRxPower", 
          "Radio_ConvertedSdrRxAgcPower", 
          "Momentum_ConvertedTotalMomentumMag"]

attdet_vars = long_df_raw_ignore[long_df_raw_ignore["variable"].str.startswith("AttDet")]["variable"].unique().tolist()

ignore_df = long_df_raw_ignore[long_df_raw_ignore['variable'].isin(attdet_vars)].reset_index(drop=True)
fill_df = long_df_raw_fill[long_df_raw_fill['variable'].isin(attdet_vars)].reset_index(drop=True)

mtad_corr_fill_30_selectedVar = mtad_analysis_by_sample(df_long=fill_df, window_size=1440, distance="Correlation")
mtad_corr_fill_30_selectedVar.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_corr_fill_30_selectedVar.csv")

mtad_eucl_fill_30_selectedVar = mtad_analysis_by_sample(df_long=fill_df, window_size=1440, distance="Euclidean")


mtad_eucl_fill_30_selectedVar.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_eucl_fill_30_selectedVar.csv")

mtad_corr_ignore_30_selectedVar = mtad_analysis_by_sample(df_long=fill_df, window_size=1440, distance="Correlation")
mtad_corr_ignore_30_selectedVar.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_corr_ignore_30_selectedVar.csv")

mtad_eucl_ignore_30_selectedVar = mtad_analysis_by_sample(df_long=fill_df, window_size=1440, distance="Euclidean")
mtad_eucl_ignore_30_selectedVar.to_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\MTAD\mtad_eucl_ignore_30_selectedVar.csv")



################ Bruce Dataset

reset = pd.DatetimeIndex([
'2023-08-11 01:30:00',
'2023-08-26 15:38:41',
'2023-09-13 17:40:00',
'2023-09-16 09:26:00',
'2023-10-28 18:40:00', 
'2024-02-25 06:22:04',
'2023-08-04 19:23:00', # operator
'2023-09-27 19:01:00', # operator
'2023-12-04 16:57:00']) # operator


import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries_with_resets(df, reset=None, k=None, start_time=None, end_time=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if start_time is not None:
        df = df[df.index >= pd.to_datetime(start_time)]
    if end_time is not None:
        df = df[df.index <= pd.to_datetime(end_time)]

    if df.empty:
        print("No data in the selected time range.")
        return

    plt.figure(figsize=(14, 6))

    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
        plt.scatter(df.index, df[col])
        if k is not None:
            std = df[col].std()
            mean = df[col].mean()
            outliers = np.abs(df[col] - mean) > k * std
            plt.scatter(df.index[outliers], df[col][outliers], color='orange', s=25, label=f'{col} > {k}×std')

    if reset is not None:
        for rt in reset:
            plt.axvline(pd.to_datetime(rt), color='r', alpha=0.3, linestyle='--', lw=2, label='Reset')

    plt.xlabel("")
    plt.ylabel("")
    plt.title("TAD on XVI Using Euclidean Distance (One Day Interval)")

    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def plot_timeseries_with_resets(df, reset=None, k=None, start_time=None, end_time=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if start_time is not None:
        df = df[df.index >= pd.to_datetime(start_time)]
    if end_time is not None:
        df = df[df.index <= pd.to_datetime(end_time)]

    if df.empty:
        print("No data in the selected time range.")
        return

    plt.figure(figsize=(14, 6))

    col = 'Distance'
    plt.plot(df.index, df[col], label=col)
    plt.scatter(df.index, df[col])

    pred_anomaly_mask = None
    if k is not None:
        std = df[col].std()
        mean = df[col].mean()
        pred_anomaly_mask = np.abs(df[col] - mean) > k * std
        plt.scatter(df.index[pred_anomaly_mask], df[col][pred_anomaly_mask], color='red', marker='x', s=50, label='Predicted Anomaly')

    if reset is not None:
        reset_times = pd.to_datetime(reset)
        for rt in reset_times:
            plt.axvline(rt, color='r', alpha=0.3, linestyle='--', lw=2, label='Reset (Real Anomaly)')

        # build binary vector for real anomalies
        real = df.index.isin(reset_times).astype(int)
    else:
        real = np.zeros(len(df), dtype=int)

    plt.xlabel("")
    plt.ylabel("")
    plt.title("TAD on XVI Using Euclidean Distance (One Day Interval)")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

    if k is not None:
        y_pred = pred_anomaly_mask.astype(int)
        y_true = real
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)




mag_file = pd.read_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Mag_evaluation\Mag1.csv")
mag_file = mag_file.set_index('time')
mag_file.index = pd.to_datetime(mag_file.index)
mag_file.shape
mag_file_clean = mag_file[~mag_file.duplicated(keep='first')]
mag_file_clean.shape

mag_file.describe
# time_deltas = mag_file_clean.index.to_series().diff().dt.total_seconds()
# time_deltas.plot(title='Sampling Gap Over Time')

print(mag_file_clean.index)
print(mag_file_clean.index.freq)
mag_file_resampled = mag_file_clean.resample('1s').mean().interpolate().fillna(method='ffill')

mag_file_resampled.index.name = "UtcTime"
mag_file_resampled.index = mag_file_resampled.index.tz_localize('UTC')

#mag_file_resampled = mag_file_resampled.reset_index() 

# mag_file_resampled = mag_file_resampled[
#     pd.to_datetime(mag_file_resampled["UtcTime"]) < pd.Timestamp("2023-10-15")
# ]

long_df_mag = transform_to_long_format_rawdata(mag_file_resampled)

mtad_eucl_mag_12 = mtad_analysis_by_sample(df_long=long_df_mag, window_size=24*60*60, distance="Euclidean")
mtad_corr_mag_12 = mtad_analysis_by_sample(df_long=long_df_mag, window_size=12*60*60, distance="Correlation")

mtad_cos_mag = mtad_analysis_by_sample(df_long=long_df_mag, window_size=12*60*30, distance="Cosine")
mtad_manhattan_mag_12 = mtad_analysis_by_sample(df_long=long_df_mag, window_size=12*60*60, distance="Manhattan")
mtad_jaccard_mag = mtad_analysis_by_sample(df_long=long_df_mag, window_size=5*60*30, distance="Jaccard") # Not working very well


plot_timeseries_with_resets(mtad_eucl_mag_12, reset=reset, k=1, start_time='2023-08-01 00:00:00', end_time='2024-04-01 00:00:00')
plot_timeseries_with_resets(mtad_corr_mag_12, reset=reset, k=1, start_time='2023-08-01 00:00:00', end_time='2024-04-01 00:00:00')


plot_timeseries_with_resets(mtad_cos_mag, reset=reset, k=2, start_time='2023-08-01 00:00:00', end_time='2024-04-01 00:00:00')
plot_timeseries_with_resets(mtad_manhattan_mag_12, reset=reset, k=2, start_time='2023-08-01 00:00:00', end_time='2024-04-01 00:00:00')
plot_timeseries_with_resets(mtad_eucl_fill, reset=reset, k=2, start_time='2023-08-01 00:00:00', end_time='2024-04-01 00:00:00')



# Compute time differences
time_deltas = mag_file_resampled.index.to_series().diff().dropna()

# Count occurrences of each delta
delta_counts = time_deltas.value_counts().sort_index()

# Convert timedelta to frequency in Hz
hz_info = pd.DataFrame({
    'delta': delta_counts.index,
    'count': delta_counts.values,
    'frequency_hz': 1 / delta_counts.index.total_seconds()
})




import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=hz_info['delta'].astype(str), y=hz_info['count'])
plt.xticks(rotation=45)
plt.title('Distribution of Time Gaps')
plt.xlabel('Time Delta')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

print(hz_info)
mag_file.columns


plot_mtad_distances(mtad_corr_fill_30)
plot_wasserstein_time_series_flag_anomalies(mtad_corr_fill_30["Time"], mtad_corr_fill_30["Distance"], reset=fault_times, k=2)


plot_mtad_distances(mtad_eucl_fill_30)
plot_wasserstein_time_series_flag_anomalies(mtad_eucl_fill_30.index, mtad_eucl_fill_30["Distance"], reset=fault_times, k=2)


plot_mtad_distances(mtad_corr_ignore_30)
plot_wasserstein_time_series_flag_anomalies(mtad_corr_ignore_30["Time"], mtad_corr_ignore_30["Distance"], reset=fault_times, k=2)


plot_mtad_distances(mtad_eucl_ignore_30)
plot_wasserstein_time_series_flag_anomalies(mtad_eucl_ignore_30["Time"], mtad_eucl_ignore_30["Distance"], reset=fault_times, k=2)


































def count_nan_in_graphs(graphs):
    total_nans = 0
    for layer_dict in graphs.values():
        for adj in layer_dict.values():
            total_nans += np.isnan(adj).sum()
    return total_nans


def count_inf_in_graphs(graphs):
    return sum(np.isinf(adj).sum() for layer in graphs.values() for adj in layer.values())

inf_count = count_inf_in_graphs(geodesic_distances)
print(f"Total ∞ values in geodesic distances: {inf_count}")


nan_count = count_nan_in_graphs(graphs)
print(f"Total NaN values in graphs: {nan_count}")

def count_nans_in_dataframe(df):
    total_nans = df.isna().sum().sum()
    nans_per_column = df.isna().sum()
    return total_nans, nans_per_column

# Usage
total_nans, nans_per_column = count_nans_in_dataframe(long_df_raw_fill)
print(f"Total NaNs in input data: {total_nans}")
print("NaNs per column:")
print(nans_per_column[nans_per_column > 0])

#from EDA.my_functions import *
import os
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.spatial.distance import cosine, correlation
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
import persim
from tqdm import tqdm

# Functions
def read_and_merge_csv(folder_path):
    """
    Reads and merges CSV files from a folder, extracting 'UtcTime' and 'prediction' columns.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.

    Returns:
    pandas.DataFrame: Merged DataFrame with 'UtcTime' as the index and predictions as columns.

    Notes:
    - Saves the merged output as 'mergedOutput.csv' in the same folder.
    """
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
    merged_df.to_csv(os.path.join(folder_path, "mergedOutput.csv"))
    print(f"Data merged and saved to {folder_path}")
    return(merged_df)


def transform_to_long_format(data):
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


def create_graphs(data, time_interval, nodes, layer, distance, threshold):
    layers = data[layer].unique()
    temporal_graphs = {}

    for l in layers:
        layer_data = data[data[layer] == l]
        layer_data = layer_data.set_index('time')
        grouped = layer_data.groupby(pd.Grouper(freq=time_interval))

        layer_graphs = {}

        for time_window, group in tqdm(grouped):
            node_values = group[nodes].unique()
            n_nodes = len(node_values)
            adjacency_matrix = np.zeros((n_nodes, n_nodes))

            for i, node_a in enumerate(node_values):
                for j, node_b in enumerate(node_values):
                    if i != j:
                        pred_a = group[group[nodes] == node_a]['prediction'].values
                        pred_b = group[group[nodes] == node_b]['prediction'].values

                        if len(pred_a) == 0 or len(pred_b) == 0:
                            continue

                        if distance == "Hamming":
                            d = np.sum(pred_a != pred_b)
                        elif distance == "Euclidean":
                            d = np.linalg.norm(pred_a - pred_b)
                        elif distance == "Jaccard":
                            pred_a = pred_a.astype(bool)
                            pred_b = pred_b.astype(bool)
                            intersection = np.sum(pred_a & pred_b)
                            union = np.sum(pred_a | pred_b)
                            d = 1 - (intersection / union) if union != 0 else 0
                        elif distance == "Manhattan":
                            d = np.sum(np.abs(pred_a - pred_b))
                        elif distance == "Cosine":
                            d = cosine(pred_a, pred_b) if np.linalg.norm(pred_a) != 0 and np.linalg.norm(pred_b) != 0 else 1
                        elif distance == "Correlation":
                            d = correlation(pred_a, pred_b)
                        elif distance == "KL":
                            pred_a = pred_a / np.sum(pred_a)
                            pred_b = pred_b / np.sum(pred_b)
                            d = np.sum(rel_entr(pred_a, pred_b))
                        else:
                            raise ValueError("Undefined Distance Metric")

                        adjacency_matrix[i, j] = d if d <= threshold else 0

            layer_graphs[time_window] = adjacency_matrix

        temporal_graphs[l] = layer_graphs

    return temporal_graphs    


def compute_geodesic_distances_scipy(temporal_graphs):
    """
    Computes geodesic distance matrices for temporal graphs using SciPy.

    Parameters:
    temporal_graphs (dict): A nested dictionary {layer -> {time_window -> adjacency_matrix}}.

    Returns:
    dict: A nested dictionary {layer -> {time_window -> distance_matrix}} with geodesic distances.

    Notes:
    - Uses SciPy's `shortest_path` function to compute pairwise shortest paths.
    - Assumes undirected and weighted graphs.
    """
  
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

    # Normalize distances
    max_distance = max(wasserstein_distances)
    normalized_distances = [dist / max_distance for dist in wasserstein_distances]

    return time_stamps, wasserstein_distances


def plot_wasserstein_time_series(time_stamps, wasserstein_distances, file_name, title="MTDA Time Series"):
    """
    Plots the Wasserstein distance time series.

    Parameters:
    time_stamps (list): List of timestamps corresponding to the distances.
    normalized_distances (list): List of computed and normalized Wasserstein distances.
    title (str, optional): Title of the plot (default: "WTDA Time Series").

    """

    # Convert timestamps to pandas datetime
    time_series_df = pd.DataFrame({"Time": time_stamps[1:], "WTDA": wasserstein_distances})

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_df["Time"],
              time_series_df["WTDA"],
                color="blue", 
                linestyle="-", 
                marker="o",
                  markersize=3)
    
    plt.xlabel("Time")
    plt.ylabel("WTDA")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    #plt.show()
    plt.savefig(fr"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Model_Integration\test\{file_name}_.png")
    plt.close()

if __name__ == "__main__":
    
    # Read and merge the data from the folder (change the path to your folder)
    df = read_and_merge_csv(folder_path=r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\outputArchive")
    df = df.dropna()
    # Transform the data to long format
    long_df = transform_to_long_format(data=df)
    long_df["model"].unique()

    # Filter the data for a specific category
    #data = long_df[long_df['category'] == "Analogs"][["time", "model", "variable", "prediction"]]
    # data = long_df[long_df['category'] == "Tracker2"][["time", "model", "variable", "prediction"]]
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
    data = long_df[long_df["model"] == "autoencoder"]
    # 1: Create temporal networks

    for i in ['Jaccard']:
        graphs = create_graphs(long_df, time_interval='24h', nodes="variable", layer="model", distance='Jaccard', threshold=0.80)
        # 2: Calculaet geodesic 
        geodesic_distances = compute_geodesic_distances_scipy(graphs)

        # 3: Generate the Persistance Diagram
        persistence_diagrams = compute_persistence_diagrams(geodesic_distances)

        # 4: SPD
        stacked_diagrams = stack_persistence_diagrams_by_time(persistence_diagrams)

        # 5: Compute the Wasserstein Distance
        time_stamps, normalized_distances = compute_wasserstein_distances(stacked_diagrams)

        # 6: Plot the time series
        plot_wasserstein_time_series(time_stamps, normalized_distances, file_name=f"{i}_threshold.80", title="")
    

    # ## Simulated data
    # data = pd.read_csv(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\Simulated_data\mergedOutput.csv")

    # long_data = (data.reset_index().melt(id_vars='UtcTime', var_name="full_column", value_name="prediction"))
    
    # # Rename the index column to 'time'
    # long_data.rename(columns={"UtcTime": "time"}, inplace=True)
    # long_data = long_data[long_data["full_column"] != "index"]
    # # Extract components from the column names
    # long_data["model"] = long_data["full_column"].str.extract(r"^(.*?)%")
    # long_data["variable"] = long_data["full_column"].str.extract(r"%(.*)$")
    # long_data['time'] = pd.to_datetime(long_data['time'])
    
    # # 1: Create temporal networks
    # graphs = create_graphs(long_data, time_interval='24h', nodes="variable", layer="model", distance="Cosine")

    # # 2: Calculaet geodesic 
    # geodesic_distances = compute_geodesic_distances_scipy(graphs)

    # # 3: Generate the Persistance Diagram
    # persistence_diagrams = compute_persistence_diagrams(geodesic_distances)

    # # 4: SPD
    # stacked_diagrams = stack_persistence_diagrams_by_time(persistence_diagrams)

    # # 5: Compute the Wasserstein Distance
    # time_stamps, normalized_distances = compute_wasserstein_distances(stacked_diagrams)

    # # 6: Plot the time series
    # plot_wasserstein_time_series(time_stamps, normalized_distances)
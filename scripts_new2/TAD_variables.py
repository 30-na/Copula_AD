from EDA.my_functions import *
from tqdm import tqdm

# Define functions for TAD algorithm
def trim_adjacency_matrix(adj, r=None, rq=.90):
    if r is None:
        upper_tri = adj[np.triu_indices_from(adj, k=1)]
        q = int(np.floor(len(upper_tri) * rq)) 
        r = np.sort(upper_tri)[q]  
        print("r: ", r)
    #print("r:", r)
    adj2 = adj.copy()
    adj2[adj > r] = 0
    return adj2, r

def construct_graph(adj, n, node_names):
    g = nx.Graph()
    g.add_nodes_from(node_names)
    # Loop over the upper triangle of the matrix to add edges
    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangular part, to avoid duplicates
            d = adj[i, j]
            if d:  # Only add edges with non-zero weight
                g.add_edge(node_names[i], node_names[j], weight=d)
    return g

def flag_anomalies(g, min_pts_bgnd, node_colors={'anomalies':'r', 'background':'b'}):
    res = {'anomalies':[],'background':[]}
    for c in nx.connected_components(g):
        if len(c) <= min_pts_bgnd:
            res['anomalies'].extend(c)
        else:
            res['background'].extend(c)
    
    for type, array in res.items():
        for node_id in array:
            g.nodes[node_id]['class'] = type
            g.nodes[node_id]['color'] = node_colors[type]
    return res, g

def calculate_anomaly_scores(classed, adj, node_names):
    scores = {}
    for a in classed['anomalies']:
        a_index = node_names.get_loc(a)
        background_index = node_names.get_indexer(classed['background'])
        
        scores[a] = np.inf  # Initialize with infinity for each anomaly
        
        for i in range(adj.shape[1]):
            if i in background_index:
                score_temp = adj[a_index, i] 
                if scores[a] > score_temp:
                    scores[a] = score_temp
    return pd.Series(scores)

def tad_classify(X, p, method='euclidean', r=None):
    if method == 'euclidean':
        # options: correlation, cosine, euclidean
        adj = squareform(pdist(X, method))
        remaining_index = X.index

    if method == "correlation":
        # too small std give us nan values
        # adj = (1 - abs(X.T.corr()).fillna(0)).values
        adj = (1 - abs(X.T.corr()).dropna(axis=0, how='all').dropna(axis=1, how='all')).values
        remaining_index = abs(X.T.corr()).dropna(axis=0, how='all').dropna(axis=1, how='all').index

    if method == 'dtw':
        # Initialize matrices for DTW distance and not_match_ratio
        n_columns = X.shape[0]
        adj = np.zeros((n_columns, n_columns))
        # Calculate DTW distance for each pair of columns
        for i in range(n_columns):
            for j in range(n_columns):
                dtw_dist = dtw.distance(X.iloc[:, i].values, X.iloc[:, j].values)
                #print(f"i = {i} and j = {j}")
                adj[i, j] = dtw_dist
        remaining_index = X.index
    
    if method == "difference":
        adj = X.dropna(axis=0, how='all').dropna(axis=1, how='all').values
        remaining_index = X.dropna(axis=0, how='all').dropna(axis=1, how='all').index

    trim_adj, r = trim_adjacency_matrix(adj, r)
    n = adj.shape[0]
    g = construct_graph(trim_adj, n, node_names=remaining_index)
    classed, g =  flag_anomalies(g=g, min_pts_bgnd=np.ceil(n*p))
    scores = calculate_anomaly_scores(classed, adj, remaining_index)
    return {'classed':classed, 'g':g, 'scores':scores, 'r':r, 'min_pts_bgnd':n*p, 'distances':adj}

def plot_heat(tad, path, metric, windows_time, vmin=None, vmax=None):
    plt.figure(figsize=(14, 12))

    sns.heatmap(tad["distances"], cmap='coolwarm', annot=True, fmt='.2f', xticklabels=tad["g"].nodes, yticklabels=tad["g"].nodes, vmin=vmin, vmax=vmax)
    plt.subplots_adjust(bottom=0.2, left=.2)
    plt.title(f'Heatmap of {metric} distance of variables for window {windows_time}')
    plt.savefig(path)
    plt.close()

def plot_graph(g, path, windows_time):
    pos = nx.spring_layout(g) 
    node_colors = [g.nodes[node]['color'] for node in g.nodes]
    nx.draw(g, pos, with_labels=True, node_color=node_colors, 
            node_size=500, edge_color='black', font_size=6)
    plt.gcf().suptitle(f"Anomalies and Background at {windows_time}")
    #plt.title("Graph Representation with Anomalies and Background")
    plt.savefig(path)
    plt.close()

def distribution_distances_list(list_data):
    #all_values = np.concatenate([dist for sublist in list_data for dist in sublist['distances']])
    #all_values = np.concatenate([matrix.flatten() for matrix in list_data])
    all_values = np.concatenate([df.to_numpy().flatten() for df in list_data])
    quintile = np.quantile(all_values, 0.95)
    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.axvline(quintile, color='red', linestyle='--', linewidth=1.5, label=f'99th Percentile: {quintile:.2f}')
    plt.xlabel("L2 Difference Values")
    plt.ylabel("Density")
    plt.title("Distribution of L2 Difference Values")
    plt.legend()  
    plt.show()
    return(quintile)
#################################

plot = 0
stnd = 1
metric = ["correlation", "euclidean", "dtw", "difference"][3]


# Choosing Variables
windows_dist_df = pd.read_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_heatmap_212.csv"), index_col=0, parse_dates=True)
variables = windows_dist_df.columns.drop("ConvertedUserAnalog5")
#variables = windows_dist_df.columns.drop(["ConvertedUserAnalog5", "ConvertedV3CurrMon"])

# Load the file and filter the variables
file = read_file_category_XVI("Analogs")
df_raw = file[variables]
# Show_TimeSeries_plot(df=df_raw, file_name = "Test", T = "", reset=None, xlab="Time", ylab="", anomaly=None, C=3)

# Standardize the data (OPTIONAL)
if stnd:
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_raw), columns=df_raw.columns)
    df_standardized.index = df_raw.index

# Resampling and Interpolation 
resample_interval = "1min"
if stnd:
    resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")
else:
    resample = df_raw[~df_raw.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")

# Divide time series data to windows
data = resample
window_size = pd.Timedelta(minutes=94)  
overlap_size = pd.Timedelta(minutes=24)  
windows = create_overlapping_windows(data, window_size, overlap_size)[:-1]
windows_time = [w.index[-1].strftime('%Y-%m-%d %H:%M:%S') for w in windows]

####################################

# Apply TAD algorithm for all windows with L2 distance

def dtw_distance_matrix(X):
    n_columns = X.shape[0]
    adj = np.zeros((n_columns, n_columns))
    # Calculate DTW distance for each pair of columns
    for i in range(n_columns):
        for j in range(i, n_columns):  # Calculate only upper triangle, assuming symmetry
            dtw_dist = dtw.distance(X.iloc[:, i].values, X.iloc[:, j].values)
            adj[i, j] = adj[j, i] = dtw_dist  # Mirror distance in symmetric matrix
    return adj


def calculate_distance_matrices(windows, method):
    distance_matrices = []
    for i, window in tqdm(enumerate(windows), desc=f"Calculating {method} matrices", total=len(windows)):
        if method == "l2":
            dist_matrix = squareform(pdist(window.T, metric='euclidean'))
        elif method == "correlation":
            dist_matrix = window.T.corr().fillna(0).values
        elif method == "dtw":
            dist_matrix = dtw_distance_matrix(window.T)
        else:
            print("The methos has not defined")
        distance_matrices.append(pd.DataFrame(dist_matrix, index=window.columns))
    
    return distance_matrices


def calculate_difference_matrices(distance_matrices):
    diff_list = []
    for i in range(1, len(distance_matrices)):
        diff_list.append(np.abs(distance_matrices[i] - distance_matrices[i - 1]))
    return diff_list


def apply_TAD_all_windows(windows, metric, resolution=None, p=0.05):
    tad_list = []
    for i, w in enumerate(windows):
        res = tad_classify(X=w.T, p=p, method=metric, r=resolution)
        tad_list.append(res)
    return tad_list


def plot_all_TAD(tad_list, plot, metric, vamx, windows_time, outputFolder = "Netl2Test"):

    output_path = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\TAD", outputFolder)

    if plot == "network":
        for i, tad in enumerate(tad_list):
            plot_graph(g=tad["g"], path=os.path.join(output_path, f"{i}.png"), windows_time=windows_time[i])
            print(i)

    if plot == "heatmap":
        for i, tad in enumerate(tad_list):
            plot_heat(tad=tad, path=os.path.join(output_path, f"{i}.png"), metric=metric, windows_time=windows_time[i], vmin=0, vmax=vamx)
            print(i)
    
    
dist_list = calculate_distance_matrices(windows, method="dtw")

diff_list = calculate_difference_matrices(dist_list)    
vmax = distribution_distances_list(list_data=diff_list)
tad_list = apply_TAD_all_windows(windows=diff_list, metric="difference", resolution=vmax)
print(tad_list[0])
plot_all_TAD(tad_list=tad_list, plot="heatmap", metric="DTW", vamx=15, windows_time=windows_time, outputFolder="HeatDTWDiff")
plot_all_TAD(tad_list=tad_list, plot="network", metric="DTW", vamx=vmax, windows_time=windows_time, outputFolder="NetDTWDiff")


###############################


import imageio
import glob
import re
# Parameters
png_dir = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\TAD\corr\netTest"
output_gif = 'corr.gif'  # Name of the output gif file
num_frames = 84  # Total number of images (0.png to 5000.png)
fps = 4  # Duration between frames in seconds (adjust as needed)

# Function to extract the numerical prefix for sorting
def extract_number(filename):
    match = re.match(r'(\d+)', filename)  # Matches the leading digits
    return int(match.group(1)) if match else float('inf')  # Return number or infinity if no match

# Get a sorted list of all PNG files in the directory
image_files = sorted(
    [f for f in os.listdir(png_dir) if f.endswith('.png')],
    key=extract_number  # Use the custom sorting function
)

for file_name in image_files:
    print(file_name)

# Read images and create GIF
with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
    for file_name in image_files:
        full_path = os.path.join(png_dir, file_name)  # Construct the full file path
        if os.path.exists(full_path):  # Check if the full file path exists
            image = imageio.imread(full_path)
            writer.append_data(image)
            print(f"Added: {full_path}")  # Log each added image
        else:
            print(f"File not found: {full_path}")  # Log missing files

print(f"GIF saved as {output_gif}")

png_dir = r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\TAD\corr\corrTest"
output_gif = 'net.gif'  # Name of the output gif file

# Function to extract the numerical prefix for sorting
def extract_number(filename):
    match = re.match(r'(\d+)', filename)  # Matches the leading digits
    return int(match.group(1)) if match else float('inf')  # Return number or infinity if no match

# Get a sorted list of all PNG files in the directory
image_files = sorted(
    [f for f in os.listdir(png_dir) if f.endswith('.png')],
    key=extract_number  # Use the custom sorting function
)

for file_name in image_files:
    print(file_name)

# Read images and create GIF
with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
    for file_name in image_files:
        full_path = os.path.join(png_dir, file_name)  # Construct the full file path
        if os.path.exists(full_path):  # Check if the full file path exists
            image = imageio.imread(full_path)
            writer.append_data(image)
            print(f"Added: {full_path}")  # Log each added image
        else:
            print(f"File not found: {full_path}")  # Log missing files

print(f"GIF saved as {output_gif}")











########## test 

df = windows[5842]
plt.figure(figsize=(14, 12))
for column in df.columns:
    plt.scatter(df.index, df[column], label=column)

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Multivariate Time Series for Original data windows 5842')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
path = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\TAD\HeatCorr\w5842.png")
plt.savefig(path)
plt.close()

# Show the plot
plt.tight_layout()
plt.show()



X = windows[0]
p = 0.05
method = 'correlation'
distances = None

if not distances:
    adj = squareform(pdist(X=windows[0].T, metric='correlation'))
    adj.shape

if not distances:
    w=windows[0]
    std = w.std(axis=0)
    w = w.replace(w.loc[:, std==0], np.nan)
    adj = 1-abs(w.corr())
    adj.shape


plt.figure(figsize=(10, 10))
sns.heatmap(adj, cmap='coolwarm', annot=True, fmt='.2f')
#plt.subplots_adjust(bottom=0.2, left=.2)
#plt.title(f'Heatmap of correlation of variables for window {windows_time[i]}')
plt.show()

trim_adj, r = trim_adjacency_matrix(adj)

plt.figure(figsize=(10, 10))
sns.heatmap(trim_adj, cmap='coolwarm', annot=True, fmt='.2f', xticklabels=variables, yticklabels=variables, vmin=0, vmax=2)
#plt.subplots_adjust(bottom=0.2, left=.2)
plt.title(f'Heatmap of correlation of variables for window {windows_time[i]}')
plt.show()

n = X.shape[0]
g = construct_graph(trim_adj, n, node_names=X.index)

classed, g =  flag_anomalies(g=g, min_pts_bgnd=np.ceil(n*p))
plot_graph(g, path= os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\corr\0.png"), windows_time="test")
scores = calculate_anomaly_scores(classed=classed, adj=adj, node_names=X.index)

g.nodes

res1 = tad_classify(X, p=0.2, method='euclidean', r=1.5)
res1["scores"]
res1["classed"]


path= os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\corr\0net.png")

pos = nx.spring_layout(g) 
nx.draw(g, pos, with_labels=True, 
        node_size=500, edge_color='black', font_size=6)
plt.gcf().suptitle(f"Anomalies and Background at {windows_time[i]}")
#plt.title("Graph Representation with Anomalies and Background")
plt.savefig(path)
plt.close()


def tad_classify(X, p, method='euclidean', r=None,  distances=None):
    if not distances:
        adj = squareform(pdist(X, method))
    trim_adj, r = trim_adjacency_matrix(adj, r=1.5)
    n = X.shape[0]
    g = construct_graph(trim_adj, n, node_names=X.columns)
    classed, g =  flag_anomalies(g=g, min_pts_bgnd=np.ceil(n*p))
    scores = calculate_anomaly_scores(classed, adj, X.columns)
    print(scores)
    return {'classed':classed, 'g':g, 'scores':scores, 'r':r, 'min_pts_bgnd':n*0.2, 'distances':adj}


########################

windows_dist_df['anomaly'] = 0
windows_dist_df['score'] = 0
row_index = res['classed']['anomalies']
windows_dist_df.iloc[row_index, windows_dist_df.columns.get_loc('anomaly')] = 1
windows_dist_df.iloc[row_index, windows_dist_df.columns.get_loc('score')] = res['scores'].get(row_index)


def plot_score(g, path):
    plt.figure(figsize=(12, 6))
    plt.plot(windows_dist_df.index, g, marker='o', linestyle='-', color='b')
    for reset_time in unknown_reset:
        plt.axvline(x=reset_time, color='r', linestyle='--', linewidth=1, label='Unknown Reset' if reset_time == unknown_reset[0] else "")
















































for w in range(1, len(windows)):
    w0 = windows[w-1]
    w1 = windows[w]  
    l2_w0 = squareform(pdist(w0.T, metric='euclidean'))
    l2_w1 = squareform(pdist(w1.T, metric='euclidean'))
    l2_dist = np.where(l2_w0 != 0, abs(l2_w0-l2_w1)/l2_w0, 0)
    l2_dist_df = pd.DataFrame(l2_dist, index=variables, columns=variables)

    # Set a threshold to filter l2 distance
    l2_threshold = pd.Series(l2_dist_df.values.flatten()).quantile(0.8)

    # Create a graph from the correlation matrix
    G = nx.Graph()

    # Add all nodes (columns of the correlation matrix) to the graph
    for col in l2_dist_df.columns:
        node = col.replace("Converted", "")
        G.add_node(node)

    # Add edges based on the correlation threshold
    for i in range(len(l2_dist_df.columns)):
        for j in range(i + 1, len(l2_dist_df.columns)):
            if abs(l2_dist_df.iloc[i, j]) > l2_threshold:
                node_i = l2_dist_df.columns[i].replace("Converted", "")
                node_j = l2_dist_df.columns[j].replace("Converted", "")
                G.add_edge(node_i, node_j, weight=l2_dist_df.iloc[i, j])

    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=5)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_size=1500, alpha=.6)  # nodes
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=.5)  # edges
    nx.draw_networkx_labels(G, pos, font_size=8)  # labels
    plt.title(f"Graph from L2 Distance Matrix for window {windows_time[w]}")
    #plt.axis('off')  # Turn off the axis
    path_dtw = os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\corr", f"N{w}.png")
    plt.savefig(path_dtw)
    plt.close()

def calculate_difference_matrix(windows, method):
    diff_list = []
    if method == "l2":
        for i in range(1, len(windows)):
            # Compute pairwise L2 distance matrices for consecutive windows
            w0 = pd.DataFrame(squareform(pdist(windows[i-1].T, metric='euclidean')))
            w1 = pd.DataFrame(squareform(pdist(windows[i].T, metric='euclidean')))
            diff_list.append(abs(l2_w0 - l2_w1))
            print(f"Processed window pair {i}/{len(windows)-1}")
    
    if method == "correlation":
        for i in range(1, len(windows)):  
            w0 = pd.DataFrame(windows[i-1]).corr()
            w1 = pd.DataFrame(windows[i]).corr()
            diff_list.append(abs(w0 - w1).fillna(0))
            print(f"Processed window pair {i}/{len(windows)-1}")
        
    if method == "dtw":
        for i in range(1, len(windows)):  
            w0 = pd.DataFrame(dtw_distance(windows[i-1]))
            w1 = pd.DataFrame(dtw_distance(windows[i]))
            diff_list.append(abs(w0 - w1))
            print(f"Processed window pair {i}/{len(windows)-1}")
    
    return diff_list

# Apply TAD algorithm for all windows with Correlation distance
tad_list_corr = []
for i, w in enumerate(windows):
    res = tad_classify(X=w.T, p=0.05, method='correlation', r=.9)
    print(i)
    tad_list_corr.append(res)


# Apply TAD algorithm for all windows with DTW distance
tad_list_dtw = []
for i, w in enumerate(windows):
    res = tad_classify(X=w.T, p=0.05, method='dtw', r=None)
    print(i)
    tad_list_dtw.append(res)
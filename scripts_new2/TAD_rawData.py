from EDA.my_functions import *


# Define functions for TAD algorithm
def trim_adjacency_matrix(adj, r=None, rq=.9):
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


def plot_graph(g, path, windows_time):
    pos = nx.spring_layout(g) 
    node_colors = [g.nodes[node]['color'] for node in g.nodes]
    nx.draw(g, pos, with_labels=True, node_color=node_colors, 
            node_size=500, edge_color='black', font_size=6)
    plt.gcf().suptitle(f"Anomalies and Background at {windows_time}")
    #plt.title("Graph Representation with Anomalies and Background")
    plt.savefig(path)
    plt.close()

def tad_classify(X, p, method='euclidean', r=None,  distances=None):
    if not distances:
        adj = squareform(pdist(X, method))
    trim_adj, r = trim_adjacency_matrix(adj, r)
    n = X.shape[0]
    g = construct_graph(trim_adj, n, node_names=X.index)
    classed, g =  flag_anomalies(g=g, min_pts_bgnd=np.ceil(n*p))
    scores = calculate_anomaly_scores(classed, adj, X.index)
    return {'classed':classed, 'g':g, 'scores':scores, 'r':r, 'min_pts_bgnd':n*p, 'distances':adj}

def plot_score(data, path):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["score"], marker='o', linestyle='-', color='b')
    #unknown_reset=pd.DatetimeIndex(['2023-08-11 01:30:00','2023-08-26 15:38:41','2023-09-13 17:40:00', '2023-09-16 09:26:00'])
    #unknown_reset = reset_times
    for reset_time in unknown_reset:
        plt.axvline(x=reset_time, color='r', linestyle='--', linewidth=1, label='Unknown Reset' if reset_time == unknown_reset[0] else "")
    #plt.axvline(x=pd.DatetimeIndex(['2023-08-11 01:30:00']), color='r', linestyle='--', linewidth=1, label='Unknown Reset')
    # Set plot title and labels
    plt.title('Anomaly scores Over Time')
    plt.xlabel('Time')
    plt.ylabel('Anomaly score')
    plt.xticks(rotation=45)
    #plt.ylim(-0.1, 1.1)  # Adjust y-limits for better visibility of 0 and 1
    plt.grid()
    # Show the plot
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#################################

# get the list of variable same as Sigma Anomaly Detection 
windows_dist_df = pd.read_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_heatmap_212.csv"), index_col=0, parse_dates=True)
variables = windows_dist_df.columns.drop("ConvertedUserAnalog5")
#variables = windows_dist_df.columns.drop(["ConvertedUserAnalog5", "ConvertedV3CurrMon"])
#load the file and filter the variables
file = read_file_category_XVI("Analogs")
df_raw = file[variables]




# Show_TimeSeries_plot(df=df_raw, file_name = "Test", T = "", reset=None, xlab="Time", ylab="", anomaly=None, C=3)
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_raw), columns=df_raw.columns)
df_standardized.index = df_raw.index


# resampling and interpolation
resample_interval = "1min"
resample = df_standardized[~df_standardized.index.duplicated(keep='first')].resample(resample_interval).mean().interpolate(method="linear")


filtered = filter_by_date(df=resample, fromDate="2023-08-01", toDate="2023-08-30")
data = resample
#data_diff = data.diff().dropna() 
data.shape

res = tad_classify(X=data, p=0.05, method='euclidean', r=2)


#data = data_diff
data['anomaly'] = 0
data['score'] = 0
row_index = res['classed']['anomalies']
data.loc[row_index, 'anomaly'] = 1
data.loc[row_index, 'score'] = res['scores'].get(row_index, 0)

plot_score(data=data, path=os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\corr", "r02min5StndFirst.png"))



plot_anomaly(g=data['anomaly'], path=os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "02.png"))







def plot_anomaly(g, path):
    plt.figure(figsize=(12, 6))
    plt.plot(windows_dist_df.index, g, marker='o', linestyle='-', color='b')
    # Set plot title and labels
    plt.title('Anomaly scores Over Time')
    plt.xlabel('Time')
    plt.ylabel('Anomaly')
    plt.xticks(rotation=45)
    #plt.ylim(-0.1, 1.1)  # Adjust y-limits for better visibility of 0 and 1
    plt.grid()
    for reset_time in unknown_reset:
        plt.axvline(x=reset_time, color='r', linestyle='--', linewidth=1, label='Unknown Reset' if reset_time == unknown_reset[0] else "")
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(path)
    plt.close()





# apply TAD algorithm for all windows 
tad_list = []
for i, corr_diff in enumerate(corr_diff_list):
    res = tad_classify(X=corr_diff, p=0.05, method='euclidean', r=1.5)
    print(i)
    tad_list.append(res)

###############
# plot the networkx (Optional)
for i, tad in enumerate(tad_list):
    plot_graph(g=tad["g"],path=os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC\corr\netRev1", f"{i}net.png"), windows_time=windows_time[i])
    print(i)

###############


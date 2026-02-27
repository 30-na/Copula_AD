from EDA.my_functions import *

from scipy.spatial.distance import pdist
 

windows_dist_df = pd.read_csv(os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Analogs_heatmap_212.csv"), index_col=0, parse_dates=True)


def trim_adjacency_matrix(adj, r=None, rq=.1):
    if r is None:
        # This is really just a lazy quantile function.
        q = int(np.floor(len(adj) * rq))
        print('q:', q)
        r = np.sort(adj)[q]
    print("r:", r)
    adj2 = adj.copy()
    adj2[adj > r] = 0
    return adj2, r


from itertools import combinations
import networkx as nx

def construct_graph(edges, n):
    g = nx.Graph()
    for z, (i, j) in enumerate(combinations(range(n), 2)):
        d = edges[z]
        if d:
            g.add_edge(i, j, weight=d)
    return g

def flag_anomalies(g, min_pts_bgnd, node_colors={'anomalies':'r', 'background':'b'}):
    res = {'anomalies':[],'background':[]}
    for c in nx.connected_components(g):
        if len(c) < min_pts_bgnd:
            res['anomalies'].extend(c)
        else:
            res['background'].extend(c)
    for type, array in res.items():
        for node_id in array:
            g.nodes[node_id]['class'] = type
            g.nodes[node_id]['color'] = node_colors[type]
    return res, g

import pandas as pd
def calculate_anomaly_scores(classed, adj, n):
    scores = {}
    for a in classed['anomalies']:
        scores[a] = 0
        for z, ij in enumerate(combinations(range(n),2)):
            i,j = ij
            if (i == a or j == a) and (
                i in classed['background'] or
                j in classed['background']):
                d = adj[z]
                if scores[a]:
                    scores[a] = np.min([scores[a], d])
                else:
                    scores[a] = d
    return pd.Series(scores)

def plot_graph(g, path):
    pos = nx.spring_layout(g) 
    node_colors = [g.nodes[node]['color'] for node in g.nodes]
    nx.draw(g, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='black', font_size=6)
    plt.title("Graph Representation with Anomalies and Background")
    plt.savefig(path)
    plt.close()

def tad_classify(X, rq, p, method='euclidean', r=None,  distances=None):
    if not distances:
        adj = pdist(X, method)
    edges, r = trim_adjacency_matrix(adj, r, rq)
    n = X.shape[0]
    g = construct_graph(edges, n)
    classed, g =  flag_anomalies(g=g, min_pts_bgnd=n*p)
    scores = calculate_anomaly_scores(classed, adj, n)
    return {'classed':classed, 'g':g, 'scores':scores, 'r':r, 'min_pts_bgnd':n*p, 'distances':adj}


res = tad_classify(X=windows_dist_df, rq=.5, p=.05)

plot_graph(res["g"], path= os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "Graph_r50_p05.png"))

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

plot_score(g=windows_dist_df['score'], path=os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "TAD_r50_p05_score.png"))
plot_anomaly(g=windows_dist_df['anomaly'], path=os.path.join(r"C:\Users\Sina.Mokhtar.XLSCIENTIFIC\Documents\Problems\XVI\SPC", "TAD_r50_p05.png"))

# Get the list of indices where anomaly is 1
anomaly_indices = windows_dist_df[windows_dist_df['anomaly'] == 1].index.tolist()
for idx in anomaly_indices:
    print(idx)

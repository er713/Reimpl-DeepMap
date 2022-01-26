from scipy.sparse import csc_matrix
import numpy as np
import networkx as nx
from . import bfs_edges, graphlet_feature_map, wl_subtree_feature_map, shortest_path_feature_map, normal, random, \
    fiedler, vertex

IMPORTANCE = {
    0: normal,
    1: vertex,
    2: fiedler,
    3: random
}


def compute_importance(adj, importance_type=0):
    n = len(adj)
    adj = adj + np.eye(n)
    cen = np.zeros(n)
    nodes = IMPORTANCE[importance_type](nx.from_numpy_matrix(adj))
    for i in range(len(nodes)):
        cen[i] = nodes[i]

    return cen


def canonicalization(graph_data, hasnl, filter_size, feature_type, graphlet_size, max_h, importance_type):
    depth = 10
    graphs = {}
    labels = {}
    attributes = {}
    num_graphs = len(graph_data[0])
    centrality_vector = {}

    num_sample = 0

    for gidx in range(num_graphs):
        # adj = graph_data[0][gidx]['am'].toarray()
        adj = graph_data[0][gidx]['am']
        n = len(adj)
        if n >= num_sample:
            num_sample = n

        graphs[gidx] = adj
        v = compute_importance(adj, importance_type)
        centrality_vector[gidx] = v

        degree = np.sum(adj, axis=1)
        if hasnl == 0:
            labels[gidx] = degree
        else:
            label = graph_data[0][gidx]['nl'].T
            labels[gidx] = label[0]

    if feature_type == 1:
        features = graphlet_feature_map(num_graphs, graphs, graphlet_size, 20)

    elif feature_type == 2:
        features = shortest_path_feature_map(num_graphs, graphs, labels)

    elif feature_type == 3:
        features = wl_subtree_feature_map(num_graphs, graphs, labels, max_h)

    else:
        raise Exception("Unknown feature type!")

    for gidx in range(num_graphs):
        path_feature = features[gidx]
        attributes[gidx] = path_feature

    # building tree-structured filters
    all_samples = {}

    for gidx in range(num_graphs):
        adj = graphs[gidx]
        nx_G = nx.from_numpy_matrix(adj)
        n = len(adj)
        cen = centrality_vector[gidx]

        sorting_vertex = -1 * np.ones(num_sample)
        cen_v = np.zeros(n)
        vertex = np.zeros(n)
        for i in range(n):
            vertex[i] = i
            cen_v[i] = cen[i]
        sub = np.argsort(cen_v)
        vertex = vertex[sub]

        if num_sample <= n:
            for i in range(num_sample):
                sorting_vertex[i] = vertex[i]

        else:
            for i in range(n):
                sorting_vertex[i] = vertex[i]

        sample = []
        for node in sorting_vertex:

            if node != -1:
                edges = list(bfs_edges(nx_G, cen, source=int(node), depth_limit=depth))
                truncated_edges = edges[:filter_size - 1]
                if not truncated_edges or len(truncated_edges) != filter_size - 1:
                    continue
                else:
                    tmp = []
                    tmp_cen = []
                    tmp.append(int(node))
                    tmp_cen.append(cen[int(node)])
                    for u, v in truncated_edges:
                        tmp.append(int(v))
                        tmp_cen.append(cen[int(v)])
                    tmp_cen = np.array(tmp_cen)
                    tmp_cen = -1 * tmp_cen
                    sub = np.argsort(tmp_cen)
                    tmp = np.array(tmp)
                    tmp = tmp[sub]
                    for v in tmp:
                        sample.append(v)
            else:
                for i in range(filter_size):
                    sample.append(-1)

        all_samples[gidx] = sample

    att = attributes[0]
    feature_size = att.shape[1]

    graph_tensor = []
    for gidx in range(num_graphs):
        sample = all_samples[gidx]
        att = attributes[gidx]
        feature_matrix = csc_matrix((num_sample * filter_size, feature_size), dtype=np.float32)
        pointer = 0
        for node in sample:
            if node != -1:
                feature_matrix[pointer, :] = att[node, :]

            pointer += 1

        graph_tensor.append(feature_matrix)

    return graph_tensor, feature_size, num_sample

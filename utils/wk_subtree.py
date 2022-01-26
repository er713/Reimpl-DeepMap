import copy
import gensim
from gensim import corpora
import networkx as nx
import numpy as np
from collections import defaultdict
from . import bfs_edges


def wl_subtree_feature_map(num_graphs, graphs, labels, max_h):
    alllabels = {}
    label_lookup = {}
    label_counter = 0
    wl_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in range(num_graphs)} for it in range(-1, max_h)}

    alllabels[0] = labels
    new_labels = {}
    # initial labeling
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        new_labels[gidx] = np.zeros(n, dtype=np.int32)
        label = labels[gidx]

        for node in range(n):
            la = label[node]
            if la not in label_lookup:
                label_lookup[la] = label_counter
                new_labels[gidx][node] = label_counter
                label_counter += 1
            else:
                new_labels[gidx][node] = label_lookup[la]
            wl_graph_map[-1][gidx][label_lookup[la]] = wl_graph_map[-1][gidx].get(label_lookup[la], 0) + 1
    compressed_labels = copy.deepcopy(new_labels)
    # WL iterations started
    for it in range(max_h - 1):
        label_lookup = {}
        label_counter = 0
        for gidx in range(num_graphs):
            adj = graphs[gidx]
            n = len(adj)
            nx_G = nx.from_numpy_matrix(adj)
            for node in range(n):
                node_label = tuple([new_labels[gidx][node]])
                neighbors = []
                edges = list(bfs_edges(nx_G, np.zeros(n), source=node, depth_limit=1))
                for u, v in edges:
                    neighbors.append(v)

                if len(neighbors) > 0:
                    neighbors_label = tuple([new_labels[gidx][i] for i in neighbors])
                    node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                if node_label not in label_lookup:
                    label_lookup[node_label] = str(label_counter)
                    compressed_labels[gidx][node] = str(label_counter)
                    label_counter += 1
                else:
                    compressed_labels[gidx][node] = label_lookup[node_label]
                wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label],
                                                                                              0) + 1
        # print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        new_labels = copy.deepcopy(compressed_labels)
        # print("labels")
        # print(labels)
        alllabels[it + 1] = new_labels

    subtrees_graph = []
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        for node in range(n):
            subtrees_node = []
            for it in range(max_h):
                graph_label = alllabels[it]
                label = graph_label[gidx]
                subtrees_node.append(str(label[node]))

            subtrees_graph.append(subtrees_node)

    dictionary = corpora.Dictionary(subtrees_graph)
    corpus = [dictionary.doc2bow(subtrees_node) for subtrees_node in subtrees_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        subtree_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = subtree_feature

    return allFeatures

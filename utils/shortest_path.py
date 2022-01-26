import gensim
import networkx as nx
from gensim import corpora
import numpy as np


def shortest_path_feature_map(num_graphs, graphs, labels):
    sp_graph = []
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        label = labels[gidx]
        nx_G = nx.from_numpy_matrix(adj)

        for i in range(n):
            sp_node = []
            for j in range(n):
                if i != j:
                    try:
                        path = list(nx.shortest_path(nx_G, i, j))
                    except nx.exception.NetworkXNoPath:
                        continue

                    if not path:
                        continue
                    if label[i] <=label[j]:
                        sp_label = str(int(label[i])) + ',' + str(int(label[j])) + ',' + str(len(path))
                    else:
                        sp_label = str(int(label[j])) + ',' + str(int(label[i])) + ',' + str(len(path))
                    sp_node.append(sp_label)
            sp_graph.append(sp_node)

    dictionary = corpora.Dictionary(sp_graph)
    corpus = [dictionary.doc2bow(sp_node) for sp_node in sp_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        sp_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = sp_feature

    return allFeatures

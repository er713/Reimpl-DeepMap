import pynauty
import pickle
import gensim
from gensim import corpora
from sklearn.preprocessing import normalize
import numpy as np


def get_graphlet(window, nsize):
    """
    This function takes the upper triangle of a nxn matrix and computes its canonical map
    """
    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i != idx] for idx, edge in enumerate(window)}

    g = pynauty.Graph(number_of_vertices=nsize, directed=False, adjacency_dict=adj_mat)
    cert = pynauty.certificate(g)
    return cert


def get_maps(n):
    # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
    file_counter = open("utils/canonical_maps/canonical_map_n%s.p" % n, "rb")
    canonical_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    # weight map -> {parent id: {child1: weight1, ...}}
    file_counter = open("utils/graphlet_counter_maps/graphlet_counter_nodebased_n%s.p" % n, "rb")
    weight_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    weight_map = {parent: {child: weight / float(sum(children.values())) for child, weight in children.items()}
                  for parent, children in weight_map.items()}
    child_map = {}
    for parent, children in weight_map.items():
        for k, v in children.items():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map


def adj_wrapper(g):
    am_ = g["al"]
    size = max(np.shape(am_))
    am = np.zeros((size, size))
    for idx, i in enumerate(am_):
        for j in i:
            am[idx][j - 1] = 1
    return am


def graphlet_feature_map(num_graphs, graphs, num_graphlets, samplesize):
    # if no graphlet is found in a graph, we will fall back to 0th graphlet of size k
    fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
    canonical_map, weight_map = get_maps(num_graphlets)
    canonical_map1, weight_map1 = get_maps(2)
    # randomly sample graphlets
    graph_map = {}
    graphlet_graph = []
    for gidx in range(num_graphs):
        # print(gidx)
        am = graphs[gidx]
        m = len(am)
        for node in range(m):
            graphlet_node = []
            for j in range(samplesize):
                rand = np.random.permutation(range(m))
                r = []
                r.append(node)
                for ele in rand:
                    if ele != node:
                        r.append(ele)

                for n in [num_graphlets]:
                    # for n in range(3,6):
                    if m >= num_graphlets:
                        window = am[np.ix_(r[0:n], r[0:n])]
                        g_type = canonical_map[get_graphlet(window, n)]
                        # for key, value in g_type.items():
                        #    print(key.decode("utf-8"))
                        #    print(value)
                        graphlet_idx = str(g_type["idx".encode()])
                    else:
                        window = am[np.ix_(r[0:2], r[0:2])]
                        g_type = canonical_map1[get_graphlet(window, 2)]
                        graphlet_idx = str(g_type["idx".encode()])

                    graphlet_node.append(graphlet_idx)

            graphlet_graph.append(graphlet_node)

    dictionary = corpora.Dictionary(graphlet_graph)
    corpus = [dictionary.doc2bow(graphlet_node) for graphlet_node in graphlet_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = normalize(M, norm='l1', axis=0)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        graphlet_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = graphlet_feature

    return allFeatures

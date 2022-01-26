import networkx as nx
import numpy as np
from sklearn.mixture import GaussianMixture


def normal(G: nx.Graph):
    return nx.eigenvector_centrality(G, max_iter=1000, tol=1.e-4)


def vertex(G: nx.Graph):
    return {i: len(v) for i, v in G.adjacency()}


def fiedler(G: nx.Graph):
    fiedler_np = np.ones(len(G), dtype=np.float)
    for i, nodes in enumerate(nx.algorithms.components.connected_components(G)):
        if len(nodes) >= 3:
            g = G.subgraph(nodes)
            fiedler = nx.fiedler_vector(g)
            m_score, m_em, m_means = -np.inf, None, None
            for n in range(1, int(np.log2(len(g))) + 1):
                em = GaussianMixture(n_components=n, max_iter=1000, tol=1.e-5)
                data = [[f] for f in fiedler]
                em.fit(data)
                score = em.score(data)
                means = [m[0] for m in em.means_]
                if m_score < score:
                    m_score, m_em, m_means = score, em, means
            fv = np.array([fiedler for _ in m_means]).T
            m = np.array(m_means)
            d = (fv - m) ** 2
            b = np.min(d, axis=1)
            fiedler_np[list(nodes)] = b
    fiedler_np = -fiedler_np + np.max(fiedler_np[fiedler_np != 1.])
    fiedler_np = fiedler_np / np.max(fiedler_np)
    return {i: s for i, s in enumerate(fiedler_np)}


def random(G: nx.Graph):
    return {i: v for i, v in enumerate(np.random.rand(len(G)))}

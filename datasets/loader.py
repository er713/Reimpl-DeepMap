from scipy.io import loadmat
import numpy as np


def load_dataset(ds_name, DATASET_DIR='./datasets/'):
    data = loadmat(f'{DATASET_DIR}{ds_name}.mat')
    graph_data = data['graph']
    graph_labels = data['label'].T[0]

    # num_graphs = len(graph_data[0])
    num_class = len(np.unique(graph_labels))
    return graph_data, graph_labels, num_class

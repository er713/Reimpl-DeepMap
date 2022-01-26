from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from . import load_dataset
from . import canonicalization


def prepare_data(graph_data, graph_labels, hasnl, filter_size, feature_type, graphlet_size, max_h, k_folds,
                 importacnce_type):
    X, feature_size, num_sample = canonicalization(graph_data, hasnl, filter_size, feature_type, graphlet_size, max_h,
                                                   importacnce_type)
    folds = list(StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=7)
                 .split(np.zeros(len(graph_data[0])), graph_labels))

    encoder = LabelEncoder()
    encoder.fit(graph_labels)
    encoded_Y = encoder.transform(graph_labels)
    Y = to_categorical(encoded_Y)

    return X, Y, folds, feature_size, num_sample


def load_prepare(ds_name, hasnl, filter_size, feature_type, importance_type, graphlet_size, max_h, k_folds,
                 DATASET_DIR='./datasets/'):
    graph_data, graph_labels, num_class = load_dataset(ds_name, DATASET_DIR)
    X, Y, folds, feature_size, num_sample = prepare_data(graph_data, graph_labels, hasnl, filter_size, feature_type,
                                                         graphlet_size, max_h, k_folds, importance_type)
    return X, Y, folds, feature_size, num_sample, num_class

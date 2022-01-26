import json
import keras
import numpy as np
import tensorflow as tf
import sys
from keras.optimizer_experimental.rmsprop import RMSprop
from keras.metrics import accuracy
from keras import backend as K
from . import load_prepare
from models import DeepMap
from losses import deepmap_loss


def learning_loop(ds_name, feature_type, importance_type, hasnl, filter_size, graphlet_size, max_h, k_folds,
                  epochs, batch_size, lr, save=True, OUTPUT_DIR='./results/', DATASET_DIR: str = './datasets/'):
    val_acc = np.zeros((k_folds, epochs))
    acc = np.zeros((k_folds, epochs))

    X, Y, folds, feature_size, num_sample, num_class = load_prepare(ds_name, hasnl, filter_size, feature_type,
                                                                    importance_type, graphlet_size, max_h, k_folds,
                                                                    DATASET_DIR=DATASET_DIR)

    for j, (train_idx, test_idx) in enumerate(folds):
        f_acc, f_val = train_loop(X, Y, train_idx, test_idx, num_sample, feature_size, num_class, batch_size, epochs,
                                  filter_size, lr)
        val_acc[j, :] = f_val
        acc[j, :] = f_acc

    if save:
        val_acc_mean = np.mean(val_acc, axis=0) * 100
        val_acc_std = np.std(val_acc, axis=0) * 100
        best_epoch = np.argmax(val_acc_mean)
        print("Average Accuracy: ")
        print("%.2f%% (+/- %.2f%%)" % (val_acc_mean[best_epoch], val_acc_std[best_epoch]))
        mean_acc = np.mean(acc, axis=0)
        for i in range(len(mean_acc)):
            print(mean_acc[i])

        with open(f'{OUTPUT_DIR}{ds_name}-{importance_type}_{feature_type}.json', 'w') as file:
            json.dump({
                'val_acc_mean': val_acc_mean,
                'val_acc_std': val_acc_std,
                'mean_acc': mean_acc,
                'k_folds': k_folds
            }, file)


def train_loop(X, Y, train_idx, test_idx, num_samle, feature_size, num_class, batch_size, epochs, filter_size, lr):
    x_val = np.array([X[id].toarray() for id in test_idx])
    y_val = Y[test_idx, :]

    x = np.array([X[id].toarray() for id in train_idx])
    y = Y[train_idx, :]

    model = DeepMap(filter_size, num_samle, feature_size, num_class)
    opt = RMSprop(learning_rate=lr)

    return __train_loop(model, tf.constant(x), tf.constant(y), tf.constant(x_val), tf.constant(y_val),
                        tf.constant(batch_size), tf.constant(epochs), opt)


# @tf.function
def __train_loop(model: keras.Model, X, Y, X_val, Y_val, batch_size, epochs, opt: RMSprop):
    val = []  # np.zeros(epochs.numpy())
    acc = []  # np.zeros(epochs.numpy())
    tmp_acc, tmp_loss = [], []
    lr = opt.learning_rate.numpy()
    m_loss = np.inf
    counter = 0
    for e in range(epochs):
        for batch_start in range(0, len(X), batch_size):
            with tf.GradientTape() as grad:
                Y_pred = model(X[batch_start:batch_start + batch_size], training=True)
                loss = tf.reduce_mean(deepmap_loss(Y[batch_start:batch_start + batch_size], Y_pred))
            gradient = grad.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradient, model.trainable_variables))
            tmp_loss.append(loss)
            tmp_acc.append(
                tf.reduce_mean(
                    accuracy(tf.argmax(Y[batch_start:batch_start + batch_size], axis=-1), tf.argmax(Y_pred, axis=-1))))

        Y_pred = model(X_val)
        v_l = tf.reduce_mean(deepmap_loss(Y_val, Y_pred))
        val.append(tf.reduce_mean(accuracy(tf.argmax(Y_val, axis=-1), tf.argmax(Y_pred, axis=-1))))
        acc.append(tf.reduce_mean(tmp_acc))
        n_loss = tf.reduce_mean(tmp_loss)
        if n_loss >= m_loss:
            counter += 1
        else:
            m_loss = n_loss
            counter = 0
        if counter >= 5:
            lr *= 0.5
            K.set_value(opt.learning_rate, lr)
            # opt.learning_rate = lr
            counter = 0
        tf.print(
            f'Epoch: {e} -- acc: {acc[e]}, loss: {n_loss} -- val_acc: {val[e]}, val_loss: {v_l} -- lr: {lr}',
            output_stream=sys.stdout)
    return acc, val

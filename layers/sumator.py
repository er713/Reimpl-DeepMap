import tensorflow as tf
from keras import Model


class Sumator(Model):

    def call(self, inputs, training=None, mask=None):
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        pass


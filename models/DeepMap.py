from . import SimpleConv1D, DenseDropout, Sumator
from keras import Model, Sequential
from keras.layers import Dense, Softmax


class DeepMap(Model):

    def __init__(self, filter_size, num_instance, feature_size, num_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Sequential([
            SimpleConv1D(32, filter_size, filter_size),
            SimpleConv1D(16, 1, 1),
            SimpleConv1D(8, 1, 1),
            Sumator(),
            DenseDropout(128),
            Dense(num_class),
            Softmax()
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def get_config(self):
        pass

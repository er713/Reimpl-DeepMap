from keras.layers import Conv1D, ReLU
from keras import Model, Sequential


class SimpleConv1D(Model):

    def __init__(self, filters=32, kernel_size=1, strides=1, padding='same', use_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias),
            ReLU()
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        pass

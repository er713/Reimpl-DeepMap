from keras.layers import Dense, Dropout, ReLU
from keras import Model, Sequential


class DenseDropout(Model):

    def __init__(self, units=128, use_bias=True, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Sequential([
            Dense(units, use_bias=use_bias),
            ReLU(),
            Dropout(rate=dropout)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def get_config(self):
        pass

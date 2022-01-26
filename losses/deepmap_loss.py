from keras.metrics import categorical_crossentropy


def loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

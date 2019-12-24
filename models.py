from tensorflow.keras import models, layers, regularizers


def get_simple_nn(words_count):
    model = models.Sequential()
    model.add(layers.Dense(
        16,
        kernel_regularizer=regularizers.l2(0.001),
        activation='relu',
        input_shape=(words_count,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(
        16,
        kernel_regularizer=regularizers.l2(0.001),
        activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    return model


# def run_lstm(words_count):
#     model = models.Sequential()
#     model.add(layers.Embedding(words_count, 32))
#     model.add(layers.LSTM(32))
#     model.add(layers.Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#
#     history = model.fit(input_train, y_train,
#                         epochs=10,
#                         batch_size=128,
#                         validation_split=0.2)

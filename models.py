from tensorflow.keras import models, layers, regularizers


def run_simple_nn(words_count, train_generator, validation_generator):
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

    history = model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        epochs=6)

    return model, history

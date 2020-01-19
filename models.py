from tensorflow.keras import models, layers, regularizers


def run_simple_nn(words_count, train_generator, validation_generator, model_file_name, epochs=6, should_load=True):
    model = None

    if should_load:
        try:
            model = models.load_model(model_file_name)
        except IOError:
            pass

        if model:
            return model, None

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

    model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

    history = model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        epochs=epochs)

    model.save(model_file_name)

    return model, history

from tensorflow.keras import models, layers, regularizers


def run_simple_nn(words_count, train_generator, validation_generator, test_generator):
    nn_file_name = 'nn'
    model = None
    try:
        model = models.load_model(nn_file_name)
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

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    history = model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        epochs=6)

    results = model.evaluate_generator(generator=test_generator)
    print('results: ', results)

    model.save(nn_file_name)

    return model, history

from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from generators import DataGenerator


max_len = 10_000
batch_size = 512
words_count = 10_000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=words_count)


# def vectorize_sequences(sequences, dimension=10_000):
#     results = np.zeros((len(sequences), dimension))
#
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#
#     return results


# encoded = to_categorical(array(train_data))
train_generator = DataGenerator(
    train_data,
    train_labels,
    classes_count=words_count,
    max_size=max_len,
    batch_size=batch_size)
validation_generator = DataGenerator(
    test_data,
    test_labels,
    classes_count=words_count,
    max_size=max_len,
    batch_size=batch_size)
# train_data_converted = vectorize_sequences(train_data)
# test_data_converted = vectorize_sequences(test_data)

# train_labels_converted = np.asarray(train_labels).astype('float32')
# test_labels_converted = np.asarray(test_labels).astype('float32')
#
# partial_train_data = train_data_converted[10_000:]
# validation_data = train_data_converted[:10_000]
#
# partial_labels = train_labels_converted[10_000:]
# validation_labels = train_labels_converted[:10_000]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(words_count,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=4,
    epochs=20)

history_dict = history.history
print(history_dict.keys())
# loss_values = history_dict['loss']
# validation_loss_values = history_dict['val_loss']
# accuracy = history_dict['accuracy']
# epochs = range(1, len(accuracy) + 1)

# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, validation_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf()
# plt.plot(epochs, accuracy, 'bo', label='Training acc')

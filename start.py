from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

words_count = 10_000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=words_count)


def vectorize_sequences(sequences, dimension=words_count):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


train_data_converted = vectorize_sequences(train_data)
test_data_converted = vectorize_sequences(test_data)

train_labels_converted = np.asarray(train_labels).astype('float32')
test_labels_converted = np.asarray(test_labels).astype('float32')

data_size = 10_000
partial_train_data = train_data_converted[data_size:]
validation_data = train_data_converted[:data_size]

partial_labels = train_labels_converted[data_size:]
validation_labels = train_labels_converted[:data_size]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(data_size,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    partial_train_data,
    partial_labels,
    epochs=20,
    batch_size=512,
    validation_data=(validation_data, validation_labels))

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

from tensorflow.keras.datasets import imdb
from tensorflow.keras import models
from tensorflow.keras import layers
from numpy import asarray
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from generators import DataGenerator

validation_size = 10_000
batch_size = 512
words_count = 10_000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=words_count)

train_generator = DataGenerator(
    train_data[validation_size:],
    train_labels[validation_size:],
    classes_count=words_count,
    batch_size=batch_size)
validation_generator = DataGenerator(
    train_data[:validation_size],
    train_labels[:validation_size],
    classes_count=words_count,
    batch_size=batch_size)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(words_count,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    epochs=4)

test_generator = DataGenerator(test_data, test_labels, classes_count=words_count, batch_size=batch_size)
results = model.evaluate_generator(generator=test_generator)
print("results: ", results)

# history_dict = history.history
# loss = history_dict['loss']
# validation_loss = history_dict['val_loss']
# accuracy = history_dict['acc']
# validation_accuracy = history_dict['val_acc']
# epochs = range(1, len(accuracy) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, validation_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf()
# plt.plot(epochs, accuracy, 'bo', label='Training acc')
# plt.plot(epochs, validation_accuracy, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
test = DataGenerator.vectorize_sequences(test_data[:1], dimension=words_count)
test2 = asarray([DataGenerator.vectorize_single(test_data[0], dimesion=words_count)])
print(model.predict(test2))

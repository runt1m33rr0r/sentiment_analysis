import sys
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

from generators import DataGenerator
from utils import text_to_data, vectorize_single

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

test_generator = DataGenerator(test_data, test_labels, classes_count=words_count, batch_size=batch_size)
results = model.evaluate_generator(generator=test_generator)
print('results: ', results)

history_dict = history.history
loss = history_dict['loss']
validation_loss = history_dict['val_loss']
accuracy = history_dict['acc']
validation_accuracy = history_dict['val_acc']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, validation_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, accuracy, 'bo', label='Training acc')
plt.plot(epochs, validation_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

if len(sys.argv) > 1:
    review_text = sys.argv[1]
    review_text_data = text_to_data(review_text, num_words=words_count)
    vectorized_review = np.array([vectorize_single(review_text_data, dimesion=words_count)])

    print('converted movie review: ', review_text_data)
    print('the sentiment of the review is: ', model.predict(vectorized_review)[0][0])

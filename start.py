import sys
import matplotlib.pyplot as plt

from generators import DataGenerator
from models import run_simple_nn
from data import ImdbData

train_size = 15_000
validation_size = 10_000
test_size = 25_000
train_start = 0
train_end = train_size - 1
validation_start = train_size
validation_end = train_size + validation_size - 1
test_start = train_size + validation_size
test_end = train_size + validation_size + test_size - 1

batch_size = 512
words_count = 10_000
imdb_data = ImdbData(words_count)

print('train portion start: ', train_start)
print('train portion end: ', train_end)
print('validation portion start: ', validation_start)
print('validation portion end: ', validation_end)
print('test portion start: ', test_start)
print('test portion end: ', test_end)

train_generator = DataGenerator(
    data=imdb_data,
    slice_start=train_start,
    slice_end=train_end,
    batch_size=batch_size)
validation_generator = DataGenerator(
    data=imdb_data,
    slice_start=validation_start,
    slice_end=validation_end,
    batch_size=batch_size)
test_generator = DataGenerator(data=imdb_data, slice_start=test_start, slice_end=test_end, batch_size=batch_size)

model, history = run_simple_nn(words_count, train_generator, validation_generator, test_generator)

if history:
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
    vectorized_review = imdb_data.text_to_data(review_text)

    print('the sentiment of the review is: ', model.predict(vectorized_review)[0][0])

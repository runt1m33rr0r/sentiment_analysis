from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from generators import DataGenerator
from numpy import asarray

words_count = 10_000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=words_count)

test = DataGenerator.vectorize_sequences(test_data[:1], dimension=words_count)

print(asarray([DataGenerator.vectorize_single(test_data[0], dimesion=words_count)]))
print(test)

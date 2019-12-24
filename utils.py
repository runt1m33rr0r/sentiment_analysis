import os
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

# data is from https://s3.amazonaws.com/text-datasets/aclImdb.zip

data_dir = './data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')


def text_from_data(data, word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])

    return decoded_review


def text_to_data(text, word_index, num_words=10_000):
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_text.split(' ')

    result = []
    for word in words:
        if word in word_index and word_index[word] <= num_words:
            result.append(word_index[word])

    return result


def read_data():
    texts = []
    labels = []

    for directory in [test_dir, train_dir]:
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(directory, label_type)
            for file_name in os.listdir(dir_name):
                if '.txt' in file_name:
                    with open(os.path.join(dir_name, file_name)) as file:
                        texts.append(file.read())

                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    # numpy_labels = np.asarray(labels)
    # numpy_texts = np.asarray(texts)
    # indices = np.arange(numpy_texts.shape[0])
    # shuffled_texts = numpy_texts[indices]
    # shuffled_labels = numpy_labels[indices]

    return texts, labels


def one_hot_encode(data, max_words=10_000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    converted_data = tokenizer.texts_to_matrix(data, mode='binary')
    word_index = tokenizer.word_index

    return converted_data, word_index


def get_imdb_data():
    (texts, labels) = read_data()
    (data, index) = one_hot_encode(texts)
    labels = np.asarray(labels).astype('float32')

    with open('./data.pickle', 'wb') as handle:
        pickle.dump((data, labels, index), handle, protocol=pickle.HIGHEST_PROTOCOL)

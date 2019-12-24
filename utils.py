import os
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from random import shuffle

# imdb data is from https://s3.amazonaws.com/text-datasets/aclImdb.zip
# twitter data is from https://www.kaggle.com/kazanova/sentiment140

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')


def read_data_files():
    texts = []

    for directory in [train_dir, test_dir]:
        for label_type in ['neg', 'pos']:
            dir_name = os.path.join(directory, label_type)
            for file_name in os.listdir(dir_name):
                if '.txt' in file_name:
                    texts.append((os.path.join(dir_name, file_name), 0 if label_type == 'neg' else 1))

    shuffle(texts)
    unpacked_texts, unpacked_labels = zip(*texts)

    return list(unpacked_texts), list(unpacked_labels)


def read_twitter_data():
    data = []

    with open('data.csv', encoding='latin-1') as file:
        for line in file:
            items = line.lower().split('","')
            evaluation = int(clean_text(items[0]))
            sentiment = 0

            if evaluation == 2:
                sentiment = 0.5
            elif evaluation == 4:
                sentiment = 1

            data.append((clean_text(items[5]), sentiment))

    shuffle(data)
    unpacked_texts, unpacked_labels = zip(*data)

    return list(unpacked_texts), list(unpacked_labels)


def save_file(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)

    return None


def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation))


class ImdbData:
    tokenizer_name = 'tokenizer.pickle'
    data_name = 'data.pickle'

    def __init__(self, num_words=10_000):
        loaded_tokenizer = load_file(ImdbData.tokenizer_name)
        loaded_data = load_file(ImdbData.data_name)

        self.num_words = num_words
        self.tokenizer = loaded_tokenizer if loaded_tokenizer else Tokenizer(num_words=num_words)
        (self.texts, self.labels) = loaded_data if loaded_data else read_data_files()

        if not loaded_tokenizer:
            self.tokenizer.fit_on_texts(self.text_generator())
            save_file(ImdbData.tokenizer_name, self.tokenizer)

        if not loaded_data:
            save_file(ImdbData.data_name, (self.texts, self.labels))

    @property
    def size(self):
        return len(self.texts)

    @property
    def word_index(self):
        return self.tokenizer.word_index

    def text_generator(self):
        for text_file in self.texts:
            with open(text_file, encoding='utf8') as file:
                yield file.read()

    def one_hot_encode(self, data):
        converted_data = self.tokenizer.texts_to_matrix(data, mode='binary')

        return converted_data

    def text_to_data(self, text):
        cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = cleaned_text.split(' ')

        return self.tokenizer.texts_to_matrix([words], mode='binary')

    def read_data(self, start_idx, end_idx):
        if start_idx < 0 or end_idx >= self.size:
            return

        opened_texts = []
        for text_file in self.texts[start_idx:end_idx + 1]:
            with open(text_file, encoding='utf8') as file:
                opened_texts.append(file.read())

        numpy_labels = np.asarray(self.labels[start_idx:end_idx + 1]).astype('float32')
        numpy_texts = np.asarray(opened_texts)
        indices = np.arange(numpy_texts.shape[0])
        shuffled_texts = numpy_texts[indices]
        shuffled_labels = numpy_labels[indices]

        encoded_data = self.one_hot_encode(shuffled_texts)

        return encoded_data, shuffled_labels


class TwitterData:
    tokenizer_name = 'tokenizer_twitter.pickle'
    data_name = 'data_twitter.pickle'

    def __init__(self, num_words=10_000):
        loaded_tokenizer = load_file(TwitterData.tokenizer_name)
        loaded_data = load_file(TwitterData.data_name)

        self.tokenizer = loaded_tokenizer if loaded_tokenizer else Tokenizer(num_words=num_words)
        (self.texts, self.labels) = loaded_data if loaded_data else read_twitter_data()

        if not loaded_tokenizer:
            self.tokenizer.fit_on_texts(self.text_generator())
            save_file(TwitterData.tokenizer_name, self.tokenizer)

        if not loaded_data:
            save_file(TwitterData.data_name, (self.texts, self.labels))

    @property
    def size(self):
        return len(self.texts)

    @property
    def word_index(self):
        return self.tokenizer.word_index

    def text_generator(self):
        for text in self.texts:
            yield text

    def one_hot_encode(self, data):
        converted_data = self.tokenizer.texts_to_matrix(data, mode='binary')

        return converted_data

    def text_to_data(self, text):
        cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = cleaned_text.split(' ')

        return self.tokenizer.texts_to_matrix([words], mode='binary')

    def read_data(self, start_idx, end_idx):
        if start_idx < 0 or end_idx >= self.size:
            return

        numpy_labels = np.asarray(self.labels[start_idx:end_idx + 1]).astype('float32')
        numpy_texts = np.asarray(self.texts[start_idx:end_idx + 1])
        indices = np.arange(numpy_texts.shape[0])
        shuffled_texts = numpy_texts[indices]
        shuffled_labels = numpy_labels[indices]

        encoded_data = self.one_hot_encode(shuffled_texts)

        return encoded_data, shuffled_labels

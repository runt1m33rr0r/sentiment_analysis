import os
from random import shuffle
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

from utils import load_file, save_file, clean_text

# imdb data is from https://s3.amazonaws.com/text-datasets/aclImdb.zip
# twitter data is from https://www.kaggle.com/kazanova/sentiment140


class BaseData:
    def __init__(self, tokenizer_file_name, data_file_name, num_words=10_000):
        loaded_tokenizer = load_file(tokenizer_file_name)
        loaded_data = load_file(data_file_name)

        self._num_words = num_words
        self._tokenizer = loaded_tokenizer if loaded_tokenizer else Tokenizer(num_words=num_words)
        (self._texts, self._labels) = loaded_data if loaded_data else self._read_file_data()

        if not loaded_tokenizer:
            self._tokenizer.fit_on_texts(self._text_generator())
            save_file(tokenizer_file_name, self._tokenizer)

        if not loaded_data:
            save_file(data_file_name, (self._texts, self._labels))

    def _read_file_data(self):
        pass

    def _text_generator(self):
        pass

    def _one_hot_encode(self, data):
        converted_data = self._tokenizer.texts_to_matrix(data, mode='binary')

        return converted_data

    def _process_data(self, data, labels):
        numpy_labels = np.asarray(labels).astype('float32')
        numpy_texts = np.asarray(data)
        indices = np.arange(numpy_texts.shape[0])
        shuffled_texts = numpy_texts[indices]
        shuffled_labels = numpy_labels[indices]

        encoded_data = self._one_hot_encode(shuffled_texts)

        return encoded_data, shuffled_labels

    @property
    def size(self):
        return len(self._texts)

    def text_to_data(self, text):
        cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = cleaned_text.split(' ')

        return self._tokenizer.texts_to_matrix([words], mode='binary')

    def read_data(self, start_idx, end_idx):
        if start_idx < 0 or end_idx >= self.size:
            raise IndexError('Data index out of range!')


class ImdbData(BaseData):
    def __init__(self, num_words=10_000):
        super().__init__('tokenizer_imdb.pickle', 'data_imdb.pickle', num_words)

    def _read_file_data(self):
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
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

    def _text_generator(self):
        for text_file in self._texts:
            with open(text_file, encoding='utf8') as file:
                yield file.read()

    def read_data(self, start_idx, end_idx):
        super().read_data(start_idx, end_idx)

        opened_texts = []
        for text_file in self._texts[start_idx:end_idx + 1]:
            with open(text_file, encoding='utf8') as file:
                opened_texts.append(file.read())

        return self._process_data(opened_texts, self._labels[start_idx:end_idx + 1])


class TwitterData(BaseData):
    def __init__(self, num_words=10_000):
        super().__init__('tokenizer_twitter.pickle', 'data_twitter.pickle', num_words)

    def _read_file_data(self):
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

    def _text_generator(self):
        for text in self._texts:
            yield text

    def read_data(self, start_idx, end_idx):
        super().read_data(start_idx, end_idx)

        return self._process_data(self._texts[start_idx:end_idx + 1], self._labels[start_idx:end_idx + 1])

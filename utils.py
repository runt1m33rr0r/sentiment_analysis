import string
from tensorflow.keras.datasets import imdb
from numpy import zeros


def text_from_data(data):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])

    return decoded_review


def text_to_data(text, num_words=10_000):
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_text.split(' ')
    word_index = imdb.get_word_index()

    result = []
    for word in words:
        if word in word_index and word_index[word] <= num_words:
            result.append(word_index[word])

    return result


def vectorize_sequences(sequences, dimension=10_000):
    results = zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


def vectorize_single(sequence, dimesion=10_000):
    result = zeros((dimesion,))
    result[sequence] = 1.

    return result

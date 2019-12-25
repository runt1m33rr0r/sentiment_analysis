import os
import string
import pickle


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

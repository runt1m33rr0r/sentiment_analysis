import os
import string
import pickle
import re


def save_file(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)

    return None


def clean_text(text):
    result = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)', ' ', text)

    return result.translate(str.maketrans('', '', string.punctuation)).lower()

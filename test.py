import string
from random import shuffle


def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation))


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

print(len(unpacked_texts), len(unpacked_labels))

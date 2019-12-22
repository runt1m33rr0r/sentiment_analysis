from tensorflow.keras.utils import Sequence, to_categorical
from numpy import floor, asarray, zeros


def vectorize_sequences(sequences, dimension=10_000):
    results = zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


class DataGenerator(Sequence):
    def __init__(self, data, labels, classes_count=10_000, batch_size=1, max_size=0):
        self.data = data
        self.labels = labels
        self.max_size = max_size
        self.batch_size = batch_size
        self.classes_count = classes_count

    def __getitem__(self, index):
        batch_start = index * self.batch_size
        batch_end = (index + 1) * self.batch_size
        extracted_data = self.data[batch_start:batch_end]
        extracted_labels = self.labels[batch_start:batch_end]

        converted_data = vectorize_sequences(extracted_data, self.classes_count)

        return converted_data, asarray(extracted_labels).astype('float32')

    def __len__(self):
        data_size = self.max_size if self.max_size > 0 else len(self.data)

        return int(floor(data_size / self.batch_size))

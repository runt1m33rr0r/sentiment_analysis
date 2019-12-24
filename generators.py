from tensorflow.keras.utils import Sequence
from numpy import floor


class DataGenerator(Sequence):
    def __init__(self, data, slice_start, slice_end, batch_size=1):
        self.data = data
        self.slice_start = slice_start
        self.slice_end = slice_end
        self.batch_size = batch_size

    def __getitem__(self, index):
        batch_start = index * self.batch_size + self.slice_start
        batch_end = (index + 1) * self.batch_size + self.slice_start
        (extracted_data, extracted_labels) = self.data.read_data(batch_start, batch_end)

        return extracted_data, extracted_labels

    def __len__(self):
        return int(floor((self.slice_end - self.slice_start + 2) / self.batch_size))

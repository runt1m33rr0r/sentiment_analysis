from tensorflow.keras.utils import Sequence
from numpy import floor


class DataGenerator(Sequence):
    def __init__(self, data, slice_start, slice_end, batch_size=1):
        self._data = data
        self._slice_start = slice_start
        self._slice_end = slice_end
        self._batch_size = batch_size

    def __getitem__(self, index):
        batch_start = index * self._batch_size + self._slice_start
        batch_end = (index + 1) * self._batch_size + self._slice_start
        (extracted_data, extracted_labels) = self._data.read_data(batch_start, batch_end)

        return extracted_data, extracted_labels

    def __len__(self):
        return int(floor((self._slice_end - self._slice_start + 1) / self._batch_size))

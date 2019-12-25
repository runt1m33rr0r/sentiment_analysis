import matplotlib.pyplot as plt

from generators import DataGenerator
from models import run_simple_nn


class Experiment:
    def __init__(
            self,
            data_class,
            model_file_name,
            should_load_model,
            train_size=15_000,
            validation_size=10_000,
            test_size=25_000,
            batch_size=512,
            words_count=10_000):
        self._model_file_name = model_file_name
        self._should_load_model = should_load_model
        self._train_start = 0
        self._train_end = train_size - 1
        self._validation_start = train_size
        self._validation_end = train_size + validation_size - 1
        self._test_start = train_size + validation_size
        self._test_end = train_size + validation_size + test_size - 1
        self._batch_size = batch_size
        self._words_count = words_count
        self._data = data_class(self._words_count)

        self._train_generator = DataGenerator(
            data=self._data,
            slice_start=self._train_start,
            slice_end=self._train_end,
            batch_size=batch_size)
        self._validation_generator = DataGenerator(
            data=self._data,
            slice_start=self._validation_start,
            slice_end=self._validation_end,
            batch_size=batch_size)
        self._test_generator = DataGenerator(
            data=self._data,
            slice_start=self._test_start,
            slice_end=self._test_end,
            batch_size=self._batch_size)

        self._model = None

    def run(self):
        self._model, history = run_simple_nn(
            self._words_count,
            self._train_generator,
            self._validation_generator,
            self._model_file_name,
            should_load=self._should_load_model)

        results = self._model.evaluate_generator(generator=self._test_generator)
        print('test results: ', results)

        if history:
            history_dict = history.history
            loss = history_dict['loss']
            validation_loss = history_dict['val_loss']
            accuracy = history_dict['acc']
            validation_accuracy = history_dict['val_acc']
            epochs = range(1, len(accuracy) + 1)

            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, validation_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.clf()
            plt.plot(epochs, accuracy, 'bo', label='Training acc')
            plt.plot(epochs, validation_accuracy, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def predict(self, input_text):
        if not self._model:
            return

        vectorized_text = self._data.text_to_data(input_text)
        return self._model.predict(vectorized_text)[0][0]

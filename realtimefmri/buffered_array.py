import numpy as np


class BufferedArray():
    def __init__(self, size, dtype='float32', buffer_size=1000):
        """An array that grows with the syntax of a list and the efficiency of an ndarray

        Attributes
        ----------
        size : int
            Number of columns in the array
        dtype : str
        buffer_size : int
        """
        super(BufferedArray, self).__init__()
        self._array = np.empty((buffer_size, size), dtype)
        self._current_size = 0

    def append(self, row):
        if self._current_size + 1 >= self._array.shape[0]:
            raise IndexError(f"Buffer size of {self._array.shape[0]} exceeded")

        self._array[self._current_size] = row
        self._current_size += 1

    def get_array(self):
        return self._array[:self._current_size]

    @property
    def shape(self):
        return self._current_size, self._array.shape[1]

    def __getitem__(self, index):
        array = self.get_array()
        return array[index]

    def __repr__(self):
        return str(self.get_array())

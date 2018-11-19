import numpy as np


class BufferedArray(object):
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
        self._array[self._current_size] = row
        self._current_size += 1

    def get_array(self):
        return self._array[:self._current_size]

    def __getitem__(self, index):
        array = self.get_array()
        return array[index]

    def __repr__(self):
        return str(self.get_array())

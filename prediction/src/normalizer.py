from copy import deepcopy

import numpy as np


class Normalizer:
    def __init__(self, data):
        self.data = deepcopy(data)

    def transform(self, data=None):
        raise NotImplementedError

    def inverse_transform(self, data=None):
        raise NotImplementedError


class NoneNormalizer(Normalizer):
    def __init__(self, data):
        super().__init__(data)

    def transform(self, data=None):
        if data is None:
            return self.data
        else:
            return data

    def inverse_transform(self, data=None):
        if data is None:
            return self.data
        else:
            return data


class StandardNormalizer(Normalizer):
    """
    Standard the input
    """

    def __init__(self, data):
        super().__init__(data)
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data=None):
        if data is None:
            return (self.data - self.mean) / self.std
        else:
            return (data - self.mean) / self.std

    def inverse_transform(self, data=None):
        if data is None:
            return self.data
        else:
            return data * self.std + self.mean

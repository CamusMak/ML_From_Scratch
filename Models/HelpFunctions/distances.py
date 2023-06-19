import numpy as np


class Distances:

    def __init__(self):
        pass

    @classmethod
    def euclidian_distance(cls, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

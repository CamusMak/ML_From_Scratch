import numpy as np


class Functions:

    def __init__(self):
        pass

    @classmethod
    def most_common(cls, array):
        labels, counts = np.unique(array, return_counts=True)
        m_common = np.argmax(counts)
        return labels[m_common]


# def most_common():
#     return None


def most_common(array):
    labels, counts = np.unique(array, return_counts=True)
    m_common = np.argmax(counts)
    return labels[m_common]

import numpy as np


class Metrics:

    def __init__(self):
        pass

    @classmethod
    def accuracy_score(cls, y_true, y_predicted):
        return np.sum(y_true == y_predicted) / len(y_true)

    @classmethod
    def RSS(cls, y_true, y_predicted):
        return np.sqrt(np.sum((y_true - y_predicted) ** 2))

    @classmethod
    def MSE(cls, y_true, y_predicted):
        return 1 / len(y_true) * (np.sqrt(np.sum((y_true - y_predicted) ** 2)))

    @classmethod
    def R_squared(cls, y_true, y_predicted):
        rss = np.sqrt(np.sum((y_true - y_predicted) ** 2))
        tss = np.sum(y_true - np.full(len(y_true), np.mean(y_true)))

        r_squared = 1 - (rss / tss)

        return r_squared



def accuracy_score(y_true, y_predicted):
    return np.sum(y_true == y_predicted) / len(y_true)


def RSS(y_true, y_predicted):
    return np.sqrt(np.sum((y_true - y_predicted) ** 2))


def MSE(y_true, y_predicted):
    return 1 / len(y_true) * (np.sqrt(np.sum((y_true - y_predicted) ** 2)))


def R_squared(y_true, y_predicted):
    rss = np.sqrt(np.sum((y_true - y_predicted) ** 2))
    tss = np.sum(y_true - np.full(len(y_true), np.mean(y_true)))
    r_squared = 1 - (rss / tss)
    return r_squared

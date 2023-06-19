import numpy as np
from Models.HelpFunctions.functions import most_common
from Models.HelpFunctions.distances import euclidian_distance


class KNNClassifier:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # calculate distances between input x and all data points of train data

        distances = [euclidian_distance(x, x_train) for x_train in self.X]

        # find the k_nearest distances

        k_nearest = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_nearest]

        # find the most common
        most_common_label = most_common(k_nearest_labels)

        return most_common_label


class KNNRegressor:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # calculate distances
        distances = [euclidian_distance(x, x_train) for x_train in self.X]

        # find k nearest values
        k_nearest = np.argsort(distances)
        k_nearest_values = [self.y[i] for i in k_nearest]

        return np.mean(k_nearest_values)

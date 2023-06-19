import numpy as np
from Models.HelpFunctions import metrics as mt
from Models.HelpFunctions.preprocessing import train_test_split


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=100):
        self.lr = learning_rate
        self.n_inter = n_iters
        self.weights = None
        self.bias = None

        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_inter):
            y_predicted = np.dot(X, self.weights) + self.bias

            # gradient descent method for optimization
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        prediction = np.dot(X, self.weights) + self.bias
        return prediction


class LogisticRegression:

    def __init__(self, learning_rage, max_iteration):
        self.lr = learning_rage
        self.max_iter = max_iteration

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, X_test):
        pass

    def score(self, X_test, y_test):
        pass

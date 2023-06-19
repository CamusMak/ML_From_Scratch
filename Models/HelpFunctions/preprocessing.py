class Preprocessing:
    def __init__(self):
        pass

    @classmethod
    def train_test_split(cls, X, y, test_size=0.2, random_state=None):
        from random import sample, seed

        seed(random_state)
        indexes = list(range(len(X)))
        test_indexes = sample(indexes, int(len(X) * test_size))
        train_indexes = [indexes[i] for i in range(len(indexes)) if i not in test_indexes]
        X_test, y_test = X[test_indexes], y[test_indexes]
        X_train, y_train = X[train_indexes], y[train_indexes]

        return X_train, X_test, y_train, y_test


def train_test_split(X, y, test_size=0.2, random_state=None):
    from random import sample, seed

    seed(random_state)
    indexes = list(range(len(X)))
    test_indexes = sample(indexes, int(len(X) * test_size))
    train_indexes = [indexes[i] for i in range(len(indexes)) if i not in test_indexes]
    X_test, y_test = X[test_indexes], y[test_indexes]
    X_train, y_train = X[train_indexes], y[train_indexes]

    return X_train, X_test, y_train, y_test

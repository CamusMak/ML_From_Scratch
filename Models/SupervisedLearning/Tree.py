import numpy as np
from collections import Counter


def entropy(y):
    # compute count of each label in total labels
    counts = np.bincount(y)

    # compute probabilities of each label
    probabilities = counts / len(y)

    # calculate entropy
    entropy_ = np.sum(probabilities * -np.log2(probabilities))

    # return entropy
    return entropy_


def accuracy(y_true, y_predicted):
    return np.sum(y_true == y_predicted) / len(y_true)


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:

    def __init__(self, min_samples_size=3, max_depth=5, n_features=None):
        self.min_samples_size = min_samples_size
        self.n_features = n_features
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = np.unique(y)

        # when to stop or
        # stopping criteria
        if (depth >= self.max_depth) or \
                (n_samples < self.min_samples_size) or \
                (len(n_labels) == 1):
            # if any of stopping criteria is reached, model creation is done
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # otherwise grow to left and right

        # find the best feature and threshold to split data into left and right parts
        # randomly select feature indexes from features
        np.random.seed(123)
        features_indexes = np.random.choice(n_features, self.n_features, replace=False)

        best_feature_index, best_threshold = self._best_criteria(X, y, features_indexes)

        # split data indexes into right and left parts
        left_indexes, right_indexes = self._split(X[:, best_feature_index], best_threshold)

        # continue growing tree to left and right sides
        left = self._grow_tree(X[left_indexes, :], y[left_indexes], depth + 1)
        right = self._grow_tree(X[right_indexes, :], y[right_indexes], depth + 1)

        # if the node is node leaf node, return node with feature, threshold, left and right child (or only one
        # if one of them is None

        return Node(best_feature_index, best_threshold, left, right)

    def _best_criteria(self, X, y, feature_indexes):

        best_information_gain = 0
        split_feature_index, split_threshold = None, None

        # iterate over all features and thresholds(all unique values of a column)
        # to find out the one whit split threshold which gives the highest information gain

        for index in feature_indexes:
            X_column = X[:, index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                information_gain = self._information_gain(X_column, y, threshold)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    split_feature_index = index
                    split_threshold = threshold

        return split_feature_index, split_threshold

    def _information_gain(self, X_column, y, threshold):

        left_indexes, right_indexes = self._split(X_column, threshold)

        if (len(left_indexes) == 0) or (len(right_indexes) == 0):
            return 0  # if left side or right side is null, nothing new information does not occurred

        # otherwise
        left_child, right_child = y[left_indexes], y[right_indexes]
        parent_node_sample_size = len(y)
        left_child_sample_size, right_child_sample_size = len(left_child), len(right_child)

        # calculate parent entropy
        parent_entropy = entropy(y)

        # calculate weighted child's entropy
        weighted_child_s_entropy = (left_child_sample_size / parent_node_sample_size) * entropy(left_child) + \
                                   (right_child_sample_size / parent_node_sample_size) * entropy(right_child)

        # calculate information gain
        information_gain = parent_entropy = weighted_child_s_entropy

        return information_gain

    def _split(self, X_column, threshold):
        left_indexes = np.argwhere(X_column < threshold).flatten()
        right_indexes = np.argwhere(X_column >= threshold).flatten()

        return left_indexes, right_indexes

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # for all data points of given X starts from root node and based on datapoint value for split feature
        # return corresponding label
        return np.array([self._traverse_tree(x, self.root) for x in X])

        pass

    def _traverse_tree(self, x, node):
        # for a single data points start from root up to last leaf based on feature values that data points has
        if node.is_leaf_node():
            return node.value

        # then, check if x data point value of current node's best feature is less than threshold value for current
        # node. If so, return traverse of left child of current node
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)

        # else if x data point has greater value for spilt feature than split threshold is,
        # return traverse of right child
        return self._traverse_tree(x, node.right)


def RMSE(y_true, y_predicted):
    return np.sum(np.sqrt((y_true - y_predicted) ** 2)) / len(y_true)


class DecisionTreeRegressor:

    def __init__(self, max_depth=5, n_features=None, min_samples_size=4):
        self.max_depth = max_depth
        self.n_feats = n_features
        self.min_samples_size = min_samples_size
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if self.n_feats is None else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if (n_samples < self.min_samples_size) or \
                (depth >= self.max_depth):

            value = y.mean()
            return Node(value=value)

        feature_indexes = np.random.choice(n_features, self.n_feats, replace=False)
        best_feature_index, best_threshold = self._best_criteria(X, y, feature_indexes)
        left_indexes, right_indexes = self._split(X[:, best_feature_index], best_threshold)

        left = self._grow_tree(X[left_indexes, :], y[left_indexes], depth + 1)
        right = self._grow_tree(X[right_indexes, :], y[right_indexes], depth + 1)

        return Node(best_feature_index, best_threshold, left, right)

    def _best_criteria(self, X, y, feature_indexes):
        min_error = 0
        best_feature_index, best_threshold = None, None

        for index in feature_indexes:
            X_column = X[:, index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                information_gain = self._information_gain(X_column, threshold, y)
                if information_gain > min_error:
                    min_error = information_gain
                    best_feature_index = index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _split(self, X_column, threshold):
        left_indexes = np.argwhere(X_column < threshold).flatten()
        right_indexes = np.argwhere(X_column >= threshold).flatten()

        return left_indexes, right_indexes

    def _information_gain(self, X_columns, threshold, y):
        left_indexes, right_indexes = self._split(X_columns, threshold)

        if (len(left_indexes) == 0) or (len(right_indexes) == 0):
            return 0

        left, right = y[left_indexes], y[right_indexes]

        y_mean_parent, y_mean_left, y_mean_right = list(map(np.mean, [y, left, right]))
        total_len, left_len, right_len = list(map(len, [y, left, right]))

        parent_error = RMSE(y, y_mean_parent)

        weighted_child_error = (left_len / total_len) * RMSE(left, y_mean_left) + \
                               (right_len / total_len) * RMSE(right, y_mean_right)

        information_gain = parent_error - weighted_child_error
        return information_gain

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):

        if node.is_leaf_node:
            print(node.value)
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    # classification
    # X, y = make_classification(1000, 10)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    #
    # model = DecisionTreeClassifier(max_depth=7)
    # model.fit(X_train, y_train)
    # prediction = model.predict(X_test)
    # print(accuracy(y_test, prediction))

    # regression
    X_reg, y_reg = make_regression(1000, 12)
    # print(X_reg.shape,y_reg.shape)
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg, y_reg, test_size=.2, random_state=123)
    # print("Done!!!")
    reg_model = DecisionTreeRegressor()
    reg_model.fit(X_r_train, y_r_train)

    print("Train: Done!!")

    prediction_reg = reg_model.predict(X_r_test)
    print("Prediction is Done!!!")

    # print(RMSE(y_r_test,prediction_reg))

    # print(prediction_reg)

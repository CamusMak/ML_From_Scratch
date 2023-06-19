import numpy as np
import matplotlib.pyplot as plt


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:

    def __init__(self, k=3, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # k is the number of centroids to use fof clustering
        # so, the number of centroids is at the same time the number
        # of cluster the model should divide data set into
        self.clusters = [[] for _ in range(self.k)]

        self.centroids = []

    # unlike supervised learning, unsupervised learning algorithms
    # doesn't have fit method, since in this case we don't have labels
    # to use the to train model

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids randomly
        random_sample_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [X[ind] for ind in random_sample_idx]

        # optimization
        for _ in range(self.max_iters):
            # create & update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_ind, cluster in enumerate(clusters):
            for sample_ind in cluster:
                labels[sample_ind] = cluster_ind
        return labels

    def _create_clusters(self, centroids):
        """
        This method creates clusters based on centroids

        """
        # clusters are list of list, where nested lists contain
        # indexes of the X values which have the shortest distance
        # from any of centroids

        clusters = [[] for _ in range(self.k)]

        for ind, data_point in enumerate(self.X):
            # find the closest centroid for all data points
            # and make cluster
            centroid_ind = self._closest_centroid(data_point, centroids)
            #
            clusters[centroid_ind].append(ind)
        return clusters

    def _closest_centroid(self, data_point, centroids):

        """
        This method calculates distances between data point and
        all centroids and return the index in which distance in the
        minimum, so, the centroid index which is the closest to
        the data point
        """
        distances = [euclidian_distance(data_point, centroids[i]) for i in range(self.k)]

        return np.argmin(distances)

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_ind, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_ind] = cluster_mean

        return centroids

    def _is_converged(self, old_centroids, current_centroids):
        return np.sum(euclidian_distance(old_centroids, current_centroids)) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point,  marker="x",color='black', linewidth=2)

        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=123)

    clusters = len(np.unique(y))

    kmeans = KMeans(k=6, plot_steps=True)
    kmeans.predict(X)

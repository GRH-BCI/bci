from collections import defaultdict

import numpy as np

from bci.eeg import EEG
from fbcca.cca import cca_scores


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, gpu=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.gpu = gpu
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        import faiss
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=self.gpu)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        return self

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


class OrderedClusterer:
    def __init__(self, clusterer, centroid_metric):
        self.clusterer = clusterer

        centroids = clusterer.cluster_centers_
        ord_ = np.array(sorted(range(centroids.shape[0]), key=lambda i: centroid_metric(centroids[i])))
        self.class_map = np.zeros_like(ord_)
        for i, c in enumerate(ord_):
            self.class_map[c] = i

    def predict(self, X):
        y = self.clusterer.predict(X)
        return self.class_map[y]


class CACC:
    def __init__(self, filter_banks=((10, 17), (20, 34), (30, 51)), n_harmonics=3, a=1.25, b=0.25,
                 n_components=3, preprocess=lambda x: x):
        self.filter_banks = filter_banks
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.coef_weights = np.power(np.arange(1, len(self.filter_banks) + 1), -a) + b
        self.preprocess = preprocess
        self.cca_coefficients = defaultdict(list)
        self.predictors = {}

    def train_start(self):
        self.cca_coefficients.clear()
        self.predictors.clear()

    def train_add(self, eeg: EEG):
        ps = self.cca_weights(eeg)
        for class_ in range(ps.shape[1]):
            self.cca_coefficients[class_].append(ps[:, class_, :])

    def _build_predictor(self, X):
        # We require that the label '0' corresponds to the null state -- that is, it must have a lower first CCA
        # coefficient in its centroid

        # c = KMeans(n_clusters=2, random_state=0).fit(X)
        c = FaissKMeans(n_clusters=2).fit(X)
        # c = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X)
        # c = Birch(n_clusters=2).fit(X)

        return OrderedClusterer(c, lambda centroid: centroid[0])

    def train_finish(self):
        for class_ in self.cca_coefficients.keys():
            coeffs = np.concatenate(self.cca_coefficients[class_], axis=0)
            self.predictors[class_] = self._build_predictor(coeffs)

    def train(self, eeg: EEG):
        self.train_start()
        self.train_add(eeg)
        self.train_finish()

    def cca_weights(self, eeg):
        eeg = self.preprocess(eeg)
        p = np.array([
            cca_scores(eeg.bandpass(fb), n_harmonics=self.n_harmonics, n_components=min(eeg.n_channels, self.n_components))
            for fb in self.filter_banks
        ])
        p = np.sum(p * self.coef_weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        return p

    def predict(self, eeg: EEG):
        assert len(self.predictors) > 0
        cca_coeffs = self.cca_weights(eeg)
        y_pred = []
        predictions = {
            class_: predictor.predict(cca_coeffs[:, class_])
            for class_, predictor in self.predictors.items()
        }
        for i in range(cca_coeffs.shape[0]):
            prediction = {class_: p[i] for class_, p in predictions.items()}
            if sum(prediction.values()) != 1:  # NaN if not exactly 1 class predicted
                y_pred.append(np.nan)
            else:
                y_pred.append(next(class_ for class_, p in prediction.items() if p))
        y_pred = np.array(y_pred)
        return y_pred

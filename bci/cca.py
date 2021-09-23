import numpy as np

from .eeg import EEG


def cca(x, y):
    # Based on github.com/stochasticresearch/depmeas
    p1, p2 = x.shape[1], y.shape[1]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    Qx, Rx = np.linalg.qr(x)
    Qy, Ry = np.linalg.qr(y)

    x_rank = np.linalg.matrix_rank(Rx)
    if x_rank == 0:
        raise ValueError('rank(x) == 0')
    elif x_rank < p1:
        Qx = Qx[:, :x_rank]

    y_rank = np.linalg.matrix_rank(Ry)
    if y_rank == 0:
        raise ValueError('rank(y) == 0')
    elif y_rank < p2:
        Qy = Qy[:, :y_rank]

    _, r, _ = np.linalg.svd(Qx.T @ Qy)
    r = np.clip(r, 0, 1)

    return r[0]


def reference(freq, n_samples, *, n_harmonics, fs):
    ref = []
    ks = np.arange(n_samples) / fs
    for i in range(n_harmonics):
        ref.append(np.sin(2 * np.pi * (i+1) * freq * ks))
        ref.append(np.cos(2 * np.pi * (i+1) * freq * ks))
    return np.array(ref).T


def cca_scores(eeg, frequencies=None, n_harmonics=3, fs=None, stimuli=None):
    frequencies = frequencies if frequencies is not None else eeg.stimuli
    fs = fs if fs is not None else eeg.fs
    X = eeg.X if isinstance(eeg, EEG) else eeg
    p = np.zeros((X.shape[0], len(frequencies)))
    for i, x in enumerate(X):
        for j, freq in enumerate(frequencies):
            ref = reference(freq, x.shape[0], n_harmonics=n_harmonics, fs=fs)
            p[i, j] = cca(x, ref)
    return p


def fbcca_scores(eeg, filter_banks, n_harmonics=3, a=1.25, b=0.25):
    weights = np.power(np.arange(1, len(filter_banks) + 1), -a) + b
    p = np.array([cca_scores(eeg.bandpass(fb), n_harmonics=n_harmonics) for fb in filter_banks])
    p = np.sum(p * weights[:, np.newaxis, np.newaxis], axis=0)
    return p


class CCA:
    def __init__(self, *, filter_banks=((10, 17), (20, 34), (30, 51)), n_harmonics=3, a=1.25, b=0.25,
                 preprocess = lambda x: x):
        self.filter_banks = filter_banks
        self.n_harmonics = n_harmonics
        self.coef_weights = np.power(np.arange(1, len(self.filter_banks)+1), -a) + b
        self.preprocess = preprocess

    def predict(self, eeg: EEG, return_weights=False):
        eeg = self.preprocess(eeg)
        p = np.array([cca_scores(eeg.bandpass(fb), n_harmonics=self.n_harmonics)
                      for fb in self.filter_banks])
        p = np.sum(p * self.coef_weights[:, np.newaxis, np.newaxis], axis=0)
        if return_weights:
            return p
        else:
            return p.argmax(axis=1)

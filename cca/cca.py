import numpy as np
import svcca

from bci.eeg import EEG


def cca(x, y, *, n=None):
    p = svcca.cca_core.get_cca_similarity(x.T, y.T, compute_coefs=False, compute_dirns=False)['cca_coef1']
    if n is None:
        return p[0]
    else:
        return p[:n]


def reference(freq, n_samples, *, n_harmonics, fs):
    ref = []
    ks = np.arange(n_samples) / fs
    for i in range(n_harmonics):
        ref.append(np.sin(2 * np.pi * (i+1) * freq * ks))
        ref.append(np.cos(2 * np.pi * (i+1) * freq * ks))
    return np.array(ref).T


def cca_scores(eeg, frequencies=None, n_harmonics=3, fs=None, n_components=None, refs=None):
    frequencies = frequencies if frequencies is not None else eeg.stimuli
    X = eeg.X if isinstance(eeg, EEG) else eeg

    if n_components is None:
        p = np.zeros((X.shape[0], len(frequencies)))
    else:
        p = np.zeros((X.shape[0], len(frequencies), n_components))

    if refs is None:
        fs = fs if fs is not None else eeg.fs
        refs = [reference(freq, X.shape[1], n_harmonics=n_harmonics, fs=fs) for freq in frequencies]

    for i, x in enumerate(X):
        for j, ref in enumerate(refs):
            p[i, j] = cca(x, ref, n=n_components)
    return p


class CCA:
    def __init__(self, *, filter_banks=((10, 17), (20, 34), (30, 51)), n_harmonics=3, a=1.25, b=0.25,
                 preprocess=lambda x: x):
        self.filter_banks = filter_banks
        self.n_harmonics = n_harmonics
        self.coef_weights = np.power(np.arange(1, len(self.filter_banks)+1), -a) + b
        self.preprocess = preprocess
        self.baseline = None

    def predict(self, eeg: EEG, return_weights=False):
        eeg = self.preprocess(eeg)
        p = np.array([cca_scores(eeg.bandpass(fb), n_harmonics=self.n_harmonics)
                      for fb in self.filter_banks])
        p = np.sum(p * self.coef_weights[:, np.newaxis, np.newaxis], axis=0)

        if self.baseline is not None:
            p = p / self.baseline

        if return_weights:
            return p
        else:
            return p.argmax(axis=1)



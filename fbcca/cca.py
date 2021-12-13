import numpy as np
import svcca

from bci.eeg import EEG


def cca(x, y, *, n=None):
    """ Compute CCA coefficients between multi-channel data `x` and `y`.

    Conceptually, the first CCA coefficient is `p0 = max corr (x*v1, y*v2)` where v1, v2 are linear transforms that make
    x and y single-dimensional (i.e. they are vectors of size x_channels and y_channels, respectively).

    Uses `svcca` internally because it's faster than `sklearn.cross_decomposition.CCA`.

    :param x: `ndarray` of shape (samples, x_channels)
    :param y: `ndarray` of shape (samples, y_channels)
    :param n: Number of CCA coefficients to return
    :return: `ndarray` if first `n` CCA coefficients. If `n` is `None`, return just the first one
    """
    p = svcca.cca_core.get_cca_similarity(x.T, y.T, compute_coefs=False, compute_dirns=False)['cca_coef1']
    if n is None:
        return p[0]
    else:
        return p[:n]


def reference(freq, n_samples, *, n_harmonics, fs):
    """
    Compute reference signal of sine and cosine waves of a given frequency and its harmonics.

    Results in a signal of shape (n_samples, n_harmonics * 2). For example, `reference(10, 100, n_harmonics=2, fs=300)`
    returns an ndarray of shape (100, 4) that contains a sine waves and cosine waves of both 10Hz and 20Hz, sampled at
    300 samples/sec.

    The reference signal includes both a sine and a cosine wave of each frequency because CCA will eventually flatten
    the reference signal to a single-dimensional array of shape (n_samples,), a weighted average of each reference
    channel, and this allows the weighted average to represent a sine wave at the given frequency of any phase,
    depending on the weights. (This works because of the identity `a*cos(x) + b*sin(x) = c*cos(x+p)` where
    `s=sgn(a)*||(a, b)||, p=arctan(-b/a)`).


    :param freq: Frequency of base signal
    :param n_samples: Length of the reference signals
    :param n_harmonics: Number of harmonics to use. `n_harmonics=1` means to use just the base frequency `freq`
    :param fs: Sampling frequency
    :return: `ndarray` of shape (n_samples, n_harmonics * 2) of reference signals
    """
    ref = []
    ks = np.arange(n_samples) / fs
    for i in range(n_harmonics):
        ref.append(np.sin(2 * np.pi * (i+1) * freq * ks))
        ref.append(np.cos(2 * np.pi * (i+1) * freq * ks))
    return np.array(ref).T


def cca_scores(eeg, frequencies=None, n_harmonics=3, fs=None, n_components=None, refs=None):
    """
    Calculates CCA coefficients between input EEG and reference signals of each given frequency.

    :param eeg: `EEG` or `ndarray` containing input EEG
    :param frequencies: Frequencies used to construct reference signals if `refs` is `None`. Defaults to `eeg.stimuli`
    :param n_harmonics: Number of harmonics to use during reference signal construction.
    :param fs: Sampling rate. Defaults to `eeg.fs`.
    :param n_components: Number of CCA coefficients to calculate. If None, just gets the first coefficient.
    :param refs: Reference signals to compare against. If None, `cca_scores` will construct reference signals from the
                 given `frequencies`, `harmonics`, and `fs`.
    :return: CCA coefficients. Shape is (samples, frequencies) if `n_components` is None, else (samples, frequencies,
             components)
    """
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


class FBCCA:
    """
    Model implementing Filter Bank Canonical Correlation Analysis (https://doi.org/10.1088/1741-2560/12/4/046008).
    """
    def __init__(self, *, filter_banks=((10, 17), (20, 34), (30, 51)), n_harmonics=3, a=1.25, b=0.25,
                 preprocess=lambda x: x):
        """ Construct FBCCA instance

        :param filter_banks: Frequency windows to bandpass filter by before applying CCA
        :param n_harmonics: Number of harmonics to use in reference signal (minimum 1)
        :param a: Coefficient determining weighted average of CCA coefficients across frequency bands
        :param b: Coefficient determining weighted average of CCA coefficients across frequency bands
        :param preprocess: Preprocessing to apply to input EEG before generating predictions
        """
        self.filter_banks = filter_banks
        self.n_harmonics = n_harmonics
        self.coef_weights = np.power(np.arange(1, len(self.filter_banks)+1), -a) + b
        self.preprocess = preprocess
        self.baseline = None

    def predict(self, eeg: EEG, return_weights=False):
        """ Get predictions from EEG data

        :param eeg: Input EEG
        :param return_weights: If True, return CCA coefficients (averaged over filter bands) instead of predictions
        :return: CCA coefficients (shape trials x classes) if `return_weights` is True, else predictions (shape trials,)
        """
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

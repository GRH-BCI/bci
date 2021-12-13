import numpy as np
import matplotlib as mpl

from bci.eeg import EEG
from fbcca.cca import reference, cca

import svcca


if __name__ == '__main__':
    mpl.use('TkAgg')

    eeg = EEG.load('C:/datasets/wearable-sensing/2021-08-19-22-00-58')

    channels = ['O1', 'O2', 'Pz', 'P3', 'P4', 'T5', 'T6']
    eeg.X = eeg.X[:, :, np.isin(eeg.montage, channels)]
    eeg = eeg.notch(60).bandpass([1, 50])

    for i_trial in range(eeg.n_trials):
        x = eeg.X[i_trial, :1200, :]
        freq = eeg.stimuli[eeg.y[i_trial]]
        y = reference(freq, x.shape[0], n_harmonics=3, fs=eeg.fs)

        sklearn_scores = cca(x, y, n=6)

        svcca_scores = svcca.cca_core.get_cca_similarity(x.T, y.T, compute_coefs=False, compute_dirns=False)['cca_coef1']

        print(sklearn_scores)
        print(svcca_scores)

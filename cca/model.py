from copy import copy
from typing import Union

import numpy as np
import optuna

from bci.eeg import EEG
from .cca import CCA


def load_model(trial: Union[optuna.trial.BaseTrial], *, window_size):
    channels = [c for c in [
        trial.suggest_categorical('Pz', ['Pz', '']),
        trial.suggest_categorical('P3', ['P3', '']),
        trial.suggest_categorical('P4', ['P4', '']),
        trial.suggest_categorical('T5', ['T5', '']),
        trial.suggest_categorical('T6', ['T6', '']),
        trial.suggest_categorical('O1', ['O1', '']),
        trial.suggest_categorical('O2', ['O2', '']),
    ] if c != '']
    if not channels:
        raise optuna.exceptions.TrialPruned()

    ref = trial.suggest_categorical('ref', ['A2', 'Cz', ''])

    filter_band = trial.suggest_uniform('fb_lo', 0.01, 10), trial.suggest_uniform('fb_hi', 40, 149)
    fb_mode = trial.suggest_categorical('fb_mode', ['a', 'b', 'c'])
    fb_start = trial.suggest_uniform('fb_start', 5, 15)
    fb_step = trial.suggest_uniform('fb_step', 1, 50)
    n_fbs = trial.suggest_int('n_fbs', 1, 5)
    if fb_mode == 'a':
        filter_banks = [(fb_start, fb_start + (i + 1) * fb_step)
                        for i in range(n_fbs)]
    elif fb_mode == 'b':
        filter_banks = [(fb_start + i * fb_step, fb_start + (i + 1) * fb_step)
                        for i in range(n_fbs)]
    elif fb_mode == 'c':
        filter_banks = [(fb_start + i * fb_step, fb_start + (n_fbs) * fb_step)
                        for i in range(n_fbs)]
    else:
        raise ValueError(f'Invalid fb_mode {fb_mode}')

    filter_banks = [
        (min(start, 120), min(stop, 120))
        for start, stop in filter_banks
        if start < 120 or stop < 120
    ]

    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)

    a = trial.suggest_uniform('a', 0.01, 2)
    b = trial.suggest_uniform('b', 0.01, 2)
    window = 0, window_size

    def preprocess(eeg: EEG):
        eeg = copy(eeg)
        if ref:
            eeg.X = eeg.X - eeg.X[:, :, eeg.montage == ref]
        eeg.X = eeg.X[:, :, np.isin(eeg.montage, channels)]
        eeg.X = eeg.X[:, int(eeg.fs * window[0]):int(eeg.fs * window[1]), :]
        eeg = eeg.notch(60).bandpass(filter_band)
        return eeg

    model = CCA(filter_banks=filter_banks,
                n_harmonics=n_harmonics, a=a, b=b,
                preprocess=preprocess)
    return model

from collections import defaultdict

import numpy as np
from cca_zoo.models import MCCA

from bci.eeg import EEG
from cca.cca import cca_scores


class MsetCCA:
    def __init__(self, filter_banks=((10, 17), (20, 34), (30, 51)), n_harmonics=3, a=1.25, b=0.25,
                 preprocess=lambda x: x):
        self.filter_banks = filter_banks
        self.n_harmonics = n_harmonics
        self.coef_weights = np.power(np.arange(1, len(self.filter_banks) + 1), -a) + b
        self.preprocess = preprocess
        self.train_eegs = defaultdict(list)
        self.references = {}

    def train_start(self):
        self.train_eegs.clear()
        self.references.clear()

    def train_add(self, eeg: EEG):
        eeg = self.preprocess(eeg)
        for y in np.unique(eeg.y):
            self.train_eegs[y].append(eeg[eeg.y == y])

    def train_finish(self):
        self.references = {
            class_: MCCA(random_state=0).fit_transform([
                x
                for eeg in self.train_eegs[class_]
                for x in eeg.X
            ])[0]
            for class_ in sorted(self.train_eegs.keys())
        }

    def train(self, eeg: EEG):
        self.train_start()
        self.train_add(eeg)
        self.train_finish()

    def cca_weights(self, eeg):
        eeg = self.preprocess(eeg)
        p = np.array([
            cca_scores(
                eeg.bandpass(fb),
                n_harmonics=self.n_harmonics,
                refs=self.references.values(),
            )
            for fb in self.filter_banks
        ])
        p = np.sum(p * self.coef_weights[:, np.newaxis, np.newaxis], axis=0)
        return p

    def predict(self, eeg: EEG):
        assert len(self.references) > 0
        p = self.cca_weights(eeg)
        return p.argmax(axis=1)

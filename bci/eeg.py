import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class EEG:
    X: np.ndarray  # shape: trials, samples, channels
    y: np.ndarray  # shape: trials,
    montage: np.ndarray
    stimuli: np.ndarray
    fs: float

    @property
    def n_trials(self):
        return self.X.shape[0]

    @property
    def n_samples(self):
        return self.X.shape[1]

    @property
    def n_channels(self):
        return self.X.shape[2]

    def bandpass(self, window):
        # b, a = signal.butter(5, window, btype='band', fs=self.fs)
        # X = signal.lfilter(b, a, self.X, axis=1)

        sos = signal.butter(5, window, btype='band', fs=self.fs, output='sos')
        X = signal.sosfiltfilt(sos, self.X, axis=1, padtype='odd')

        return EEG(X, self.y, self.montage, self.stimuli, self.fs)

    def notch(self, freq):
        b, a = signal.iirnotch(freq, 30, self.fs)
        X = signal.lfilter(b, a, self.X, axis=1)

        # b, a = signal.iirnotch(60, 20, self.fs)
        # X = signal.filtfilt(b, a, self.X, axis=1, padtype='odd')

        return EEG(X, self.y, self.montage, self.stimuli, self.fs)

    @classmethod
    def load(cls, path, *, fs=300, epoch_start=0, epoch_length=8):
        path = Path(path)
        epoch_size = int(epoch_length * fs)

        data_table = pd.read_table(path / 'data.csv', sep=',', dtype=float, on_bad_lines='skip', engine='python')
        data_table.rename(columns=lambda s: s.strip(), inplace=True)
        markers_table = pd.read_table(path / 'markers.csv', sep=', ', engine='python')

        montage = data_table.keys()
        montage = montage[~montage.isin(['timestamp', 'TRG'])]

        class_names = np.array(['left', 'right', 'top', 'bottom'])
        class_stimuli = np.loadtxt(path / 'frequencies.txt', delimiter=',')

        X, y = [], []
        for _, (timestamp, marker) in markers_table.iterrows():
            data = data_table[data_table.timestamp >= timestamp]
            data = data.iloc[epoch_start:epoch_start + epoch_size]

            if data.shape[0] < epoch_size:
                print(f'Skipping marker={marker}', file=sys.stderr)
                continue

            data = data[montage]

            X.append(data)
            y.append(np.where(class_names == marker)[0][0])

        return cls(np.array(X), np.array(y), montage, class_stimuli, fs)

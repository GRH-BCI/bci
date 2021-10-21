from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class EEG:
    X: np.ndarray  # shape: trials, samples, channels
    y: np.ndarray  # shape: trials,
    montage: np.ndarray
    stimuli: np.ndarray
    fs: float

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

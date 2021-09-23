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
        b, a = signal.butter(5, window, btype='band', fs=self.fs)
        X = signal.lfilter(b, a, self.X, axis=1)
        return EEG(X, self.y, self.montage, self.stimuli, self.fs)

    def notch(self, freq):
        b, a = signal.iirnotch(freq, 30, self.fs)
        X = signal.lfilter(b, a, self.X, axis=1)
        return EEG(X, self.y, self.montage, self.stimuli, self.fs)

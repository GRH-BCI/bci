from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from bci.eeg import EEG
from fbcca.cca import reference

dataset = Path('C:/datasets/wearable-sensing')
eeg = EEG.load(dataset/'2021-07-07-17-28-18')

channels = ['O1', 'O2', 'P3', 'P4']
eeg.X = eeg.X[:, :, np.isin(eeg.montage, channels)]
eeg = eeg.notch(60).bandpass([5, 40])

t = np.arange(eeg.fs * 3) / eeg.fs

plt.style.use('dark_background')
plt.figure(figsize=(19.5, 21.52))

plt.subplot(3, 1, 1)
plt.plot(t, signal.square(2 * np.pi * 11 * t))
plt.xlabel('Time (s)')
plt.ylabel('11 Hz Stimulus')
# plt.grid()
plt.xlim(0, 1)
plt.ylim(-1.5, 1.5)

plt.subplot(3, 1, 2)
plt.plot(t, reference(11, t.shape[0], n_harmonics=2, fs=eeg.fs))
plt.xlabel('Time (s)')
plt.ylabel('11 Hz Reference')
# plt.grid()
plt.xlim(0, 1)
plt.ylim(-1.5, 1.5)

plt.subplot(3, 1, 3)
x = eeg.X[0, :t.shape[0]]
x = (x - x.mean(axis=0)) / x.std(axis=0)
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('11 Hz EEG Response')
# plt.grid()
plt.xlim(0, 1)
plt.ylim(-1.5, 1.5)

path = Path('report/figures')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path/'title_eeg.jpg', bbox_inches='tight')

plt.show()

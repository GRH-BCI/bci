import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import List

import numpy as np

from .app import App
from .eeg import EEG


class InputListener:
    def ingest_data(self, timestamp: float, eeg: EEG):
        pass

    def ingest_marker(self, timestamp: float, marker: str):
        pass


class FileRecorder(InputListener):
    def __init__(self, path: Path, channel_names: List):
        self.path = path
        self.data_file = open(self.path / 'data.csv', 'w')
        self.markers_file = open(self.path / 'markers.csv', 'w')
        self.data_file.write(f'timestamp, {", ".join(channel_names)}\n')
        self.markers_file.write('timestamp, marker\n')
        self.data_buffer = []
        self.markers_buffer = []

    def ingest_data(self, timestamp, eeg):
        self.data_buffer.append((timestamp, eeg))

        self.data_file.flush()

    def ingest_marker(self, timestamp, marker):
        self.markers_buffer.append((timestamp, marker))
        self.markers_file.write(f'{timestamp}, {marker}\n')
        self.markers_file.flush()

    def loop(self):
        while True:

            if len(self.data_buffer) > 100:
                for timestamp, eeg in self.data_buffer:
                    self.data_file.write(f'{timestamp}, {", ".join(str(d) for d in eeg.X.flatten())}\n')
                self.data_file.flush()

            if len(self.markers_buffer) > 0:
                for timestamp, marker in self.markers_buffer:
                    self.markers_file.write(f'{timestamp}, {marker}\n')
                self.markers_file.flush()

            time.sleep(0.1)


class DisconnectError(Exception):
    pass


class RealtimeModel(InputListener):
    def __init__(self, model, *, window_size, n_preds, preds_per_sec,
                 throw_on_disconnect=True):
        self.model = model
        self.window_size = window_size
        self.y_preds = deque(maxlen=n_preds)
        self.preds_per_sec = preds_per_sec
        self.lock = Lock()
        self.sample_counter = 0
        self.throw_on_disconnect = throw_on_disconnect
        self.eegs = None  # type: deque
        self.last_data_time = None

    def clear_buffers(self):
        if self.eegs is not None:
            self.eegs.clear()
        self.y_preds.clear()
        self.sample_counter = 0

    def ingest_data(self, _timestamp, eeg):
        self.last_data_time = time.time()

        if self.eegs is None:
            self.eegs = deque(maxlen=int(self.window_size * eeg.fs))

        with self.lock:
            self.eegs.append(eeg)
        self.sample_counter += 1

        if len(self.eegs) < self.eegs.maxlen:
            return  # Don't do anything until we have enough data

        pred_delay_samples = eeg.fs / self.preds_per_sec
        if self.sample_counter % pred_delay_samples == 0:
            with self.lock:
                eegs = list(self.eegs)
            eeg = EEG(
                X=np.concatenate([eeg.X for eeg in eegs], axis=1),
                y=None,
                montage=eegs[0].montage,
                stimuli=eegs[0].stimuli,
                fs=eegs[0].fs
            )
            try:
                self.y_preds.append(self.model.predict(eeg)[0])
            except ValueError as e:
                print(e, file=sys.stderr)
                self.y_preds.append(None)

    def predict(self, timeout=None):
        start_time = time.time()
        self.last_data_time = time.time()
        while True:
            y_pred = self._try_predict()
            if y_pred is not None:
                return y_pred
            if timeout is not None and time.time() > start_time + timeout:
                raise TimeoutError()
            if self.throw_on_disconnect and time.time() > self.last_data_time + 10:
                raise DisconnectError()
            time.sleep(0.1)

    def _try_predict(self):
        with self.lock:
            y_preds = list(self.y_preds)

        if len(y_preds) < self.y_preds.maxlen:
            return None

        if not all(y_preds[0] == y_pred for y_pred in y_preds):
            return None

        return y_preds[0]


def collect_input(app: App, listeners: List[InputListener]):
    while not app.dsi_input.is_attached():
        time.sleep(0.1)

    while True:

        for timestamp, data in app.dsi_input.pop_all_data():
            eeg = EEG(
                np.array(data)[np.newaxis, np.newaxis],
                None,
                np.array(app.dsi_input.get_channel_names()),
                app.freq,
                app.dsi_input.get_sampling_rate()
            )
            for l in listeners:
                l.ingest_data(timestamp, eeg)

        for timestamp, marker in app.dsi_input.pop_all_markers():
            for l in listeners:
                l.ingest_marker(timestamp, marker)

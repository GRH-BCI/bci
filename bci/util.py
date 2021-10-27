import sys
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty, LifoQueue
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
        self.data_buffer = Queue()
        self.markers_buffer = Queue()

    def ingest_data(self, timestamp, eeg):
        self.data_buffer.put((timestamp, eeg))

    def ingest_marker(self, timestamp, marker):
        self.markers_buffer.put((timestamp, marker))

    def loop(self):
        while True:

            if self.data_buffer.qsize() > 100:
                try:
                    while True:
                        timestamp, eeg = self.data_buffer.get_nowait()
                        self.data_file.write(f'{timestamp}, {", ".join(str(d) for d in eeg.X.flatten())}\n')
                except Empty:
                    pass
                self.data_file.flush()

            if self.markers_buffer.qsize() > 0:
                try:
                    while True:
                        timestamp, marker = self.markers_buffer.get_nowait()
                        self.markers_file.write(f'{timestamp}, {marker}\n')
                except Empty:
                    pass
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
        self.last_timestamp = None

    def clear_buffers(self):
        if self.eegs is not None:
            self.eegs.clear()
        self.y_preds.clear()
        self.sample_counter = 0

    def ingest_data(self, timestamp, eeg):
        # XXX: Probably shouldn't be doing computation-heavy stuff in a callback from the
        # input-distribution thread

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
        while True:
            y_pred = self._try_predict()
            if y_pred is not None:
                return y_pred
            if timeout is not None and time.time() > start_time + timeout:
                raise TimeoutError()

            time.sleep(0.1)

    def _try_predict(self):
        with self.lock:
            y_preds = list(self.y_preds)

        if len(y_preds) < self.y_preds.maxlen:
            return None

        if not all(y_preds[0] == y_pred for y_pred in y_preds):
            return None

        return y_preds[0]


class InputDistributor:
    def __init__(self, app: App, listeners: List[InputListener], *, disconnect_timeout=1):
        self.app = app
        self.listeners = listeners
        self.disconnect_timeout = disconnect_timeout

    def wait_for_connection(self):
        while True:
            for _ in self.app.dsi_input.pop_all_data():
                return
            time.sleep(0.1)

    def loop(self):
        last_data_time = time.time()

        while True:

            for timestamp, data in self.app.dsi_input.pop_all_data():
                last_data_time = time.time()
                eeg = EEG(
                    np.array(data)[np.newaxis, np.newaxis],
                    None,
                    np.array(self.app.dsi_input.get_channel_names()),
                    self.app.freq,
                    self.app.dsi_input.get_sampling_rate()
                )
                for l in self.listeners:
                    l.ingest_data(timestamp, eeg)

            for timestamp, marker in self.app.dsi_input.pop_all_markers():
                last_data_time = time.time()
                for l in self.listeners:
                    l.ingest_marker(timestamp, marker)

            if self.disconnect_timeout is not None and time.time() - last_data_time > self.disconnect_timeout:
                raise DisconnectError()

            time.sleep(0.1)

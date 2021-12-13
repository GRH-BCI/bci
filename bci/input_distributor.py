import sys
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from threading import Lock
from typing import List

import numpy as np

from .app import App
from .eeg import EEG


class InputListener:
    """
    Base class for anything that receives data/marker streams from `InputDistributor`.
    """
    def ingest_data(self, timestamp: float, eeg: EEG):
        """ Receive EEG data corresponding to a single timestamp.

        :param timestamp: timestamp of input EEG data.
        :param eeg: EEG instance with n_trials=1, n_samples=1
        """
        pass

    def ingest_marker(self, timestamp: float, marker: str):
        """ Ingests a single marker. """
        pass


class FileRecorder(InputListener):
    """
    Receives EEG data from an InputDistributor and saves it to disk in a format that can be loaded with `EEG.load()`.

    The intended use case involves three threads: the input-distributor thread, on which `InputDistributor` collects
    and distributes its EEG input; the file-recorder thread, on which `FileRecorder.loop()` runs; and the main thread,
    which orchestrates the other two and eventually calls `FileRecorder.kill()` and `InputDistributor.kill()`.
    """
    def __init__(self, path: Path, channel_names: List):
        """ Create a FileRecorder instance.

        :param path: directory to save the EEG data to.
        :param channel_names: names corresponding to channels given by the InputDistributor.
        """
        self.path = path
        self.data_file = open(self.path / 'data.csv', 'w')
        self.markers_file = open(self.path / 'markers.csv', 'w')
        self.data_file.write(f'timestamp, {", ".join(channel_names)}\n')
        self.markers_file.write('timestamp, marker\n')
        self.data_buffer = Queue()
        self.markers_buffer = Queue()
        self.done = False

    def kill(self):
        """ Terminate `loop()` when buffers  """
        self.done = True

    def ingest_data(self, timestamp, eeg):
        """ Ingests EEG from the InputDistributor. """
        if not self.done:
            self.data_buffer.put((timestamp, eeg))

    def ingest_marker(self, timestamp, marker):
        """ Ingests markers from the InputDistributor. """
        if not self.done:
            self.markers_buffer.put((timestamp, marker))

    def loop(self):
        """ Loop until killed, writing ingested data and markers to disk.
        Data is written in batches of 100 to reduce I/O usage.
        """
        while not self.data_buffer.empty() or not self.markers_buffer.empty() or not self.done:
            if self.data_buffer.qsize() > 100 or (not self.data_buffer.empty() and self.done):
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
    """
    Repeatedly runs a model on realtime EEG coming from an `InputDistributor` instance.

    Stores a sliding window of input EEG. A fixed number of times a second, runs a given model on the sliding window,
    storing the resulting prediction.

    Will not output predictions until the prediction has *stabilized*--that is, until the last `n_preds` predictions
    are the same.

    The intended use case involves two threads: the input-distributor thread, on which `InputDistributor` collects
    and distributes its EEG input, and the main thread, which calls `RealtimeModel.predict()`.
    """
    def __init__(self, model, *, window_size, n_preds, preds_per_sec):
        """

        :param model: Model to run on input EEG
        :param window_size: Size of the sliding window, in seconds
        :param n_preds: Number of predictions that must be equal for the prediction to be considered stable
        :param preds_per_sec: Number of times per second to run the model
        """
        self.model = model
        self.window_size = window_size
        self.y_preds = deque(maxlen=n_preds)
        self.preds_per_sec = preds_per_sec
        self.lock = Lock()
        self.sample_counter = 0
        self.eegs = None  # type: deque

    def clear_buffers(self):
        """
        Empty sliding window and previous prediction buffers. Can be useful if you know the previous data/predictions
        are wrong/irrelevant, e.g. when starting a new trial.
        """
        if self.eegs is not None:
            self.eegs.clear()
        self.y_preds.clear()
        self.sample_counter = 0

    def ingest_data(self, _timestamp, eeg):
        """
        Ingests EEG from InputDistributor.

        Currently this runs the model as well, which makes RealtimeModel easier to implement, but is not really a good
        idea since it will slow down the input-distributor thread.
        """
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
        """ Wait for and return a stable prediction.

        :param timeout: How long to wait until throwing TimeoutError(). If None, wait forever
        :return: Predicted classification of input EEG
        """
        start_time = time.time()
        while True:
            y_pred = self._try_predict()
            if y_pred is not None:
                return y_pred
            if timeout is not None and time.time() >= start_time + timeout:
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

    def test(self, eeg: EEG, *, return_='acc'):
        """
        Test offline on given EEG. For each trial, simulates ingesting the data, repeatedly running the model, and
        waiting for a prediction. If no stable prediction is found for a trial, the corresponding y_pred is set to NaN.

        :param eeg: Offline EEG to test the model on.
        :param return_: One of 'acc' or 'y_pred'
        :return: Mean accuracy over trials (compared against EEG.y) if return_ is 'acc', else an ndarray of predictions
        """
        ys_pred = []
        for i_trial in range(eeg.n_trials):
            self.clear_buffers()
            # print(f'{i_trial} / {eeg.n_trials}')
            y_pred = np.nan
            for i_sample in range(eeg.n_samples):
                # print(f'{i_sample} // {eeg.n_samples}')
                eeg_i = EEG(X=eeg.X[i_trial:i_trial+1, i_sample:i_sample+1], y=eeg.y[i_trial:i_trial+1], montage=eeg.montage, stimuli=eeg.stimuli, fs=eeg.fs)
                self.ingest_data(None, eeg_i)
                try:
                     y_pred = self.predict(timeout=0)
                     break
                except TimeoutError:
                    pass
            ys_pred.append(y_pred)

        # print(ys_pred, eeg.y)
        if return_ == 'acc':
            return (ys_pred == eeg.y).astype(float).mean()
        elif return_ == 'y_pred':
            return np.array(ys_pred)
        else:
            raise ValueError()


class InputDistributor:
    """
    Receives SSVEP input from a DSIInput instance and distributes it to various listeners.

    The intended use case involves two threads: the input-distributor thread where `InputDistributor.loop()` is called,
    and the main thread, which calls orchestrates the input-distributor thread, and eventually kills it by calling
    `InputDistributor.kill()`.
    """
    def __init__(self, app: App, listeners: List[InputListener], *, disconnect_timeout=1):
        """

        :param app: App to get DSIInput instance from
        :param listeners: List of listeners to give new EEG samples to through their `ingest_data()` and
                          `ingest_marker()` methods.
        :param disconnect_timeout: Print warning if the time between received batches is larger than this. Set to None
                                   to disable.
        """
        self.app = app
        self.listeners = listeners
        self.disconnect_timeout = disconnect_timeout
        self.done = False

    def kill(self):
        """ Mark the InputDistributor as killed. """
        self.done = True

    def wait_for_connection(self):
        """ Wait until the headset starts producing data """
        while not self.done:
            for _ in self.app.dsi_input.pop_all_data():
                return
            time.sleep(0.1)

    def loop(self):
        """
        Repeatedly pops data and markers off of the DSIInput queues and pushes them to all listeners.
        Stops only when the InputDistributor is marked as killed (i.e. when `kill()` is called in a separate thread).
        """
        self.wait_for_connection()
        channel_names = np.array(self.app.dsi_input.get_channel_names())

        last_data_time = time.time()

        while not self.done:

            for timestamp, data in self.app.dsi_input.pop_all_data():
                last_data_time = time.time()
                eeg = EEG(
                    np.array(data)[np.newaxis, np.newaxis],
                    None,
                    channel_names,
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
                print('Headset connection timed out')
                # raise DisconnectError()

            time.sleep(0.1)



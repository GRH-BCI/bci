import json
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Thread

import numpy as np

from bci.app import App
from bci.input_distributor import FileRecorder, InputDistributor


def main(app: App, *, path: Path, **kwargs):
    app.connect_to_headset()
    app.connect_to_leds()
    app.calibrate_leds()

    file_logger = FileRecorder(path, app.dsi_input.get_channel_names())
    input_distributor = InputDistributor(app, listeners=[file_logger])
    input_distributor.wait_for_connection()

    Thread(target=lambda: input_distributor.loop()).start()
    Thread(target=lambda: file_logger.loop()).start()

    experiment(app, **kwargs)


def experiment(app: App, window_size: float, n_repeats: int):
    app.leds.start([0, 1, 2, 3])

    app.gui.set_text('Tap to start')
    app.gui.wait_for_click()
    app.gui.set_text('')

    ys_true = np.repeat([0, 1, 2, 3], n_repeats).astype(int)
    np.random.shuffle(ys_true)

    for y_true in ys_true:
        dir_true = ['left', 'right', 'top', 'bottom'][y_true]
        app.gui.set_arrow(dir_true)
        time.sleep(0.5)  # Allow some reaction time
        timestamp = app.dsi_input.get_latest_timestamp()
        app.dsi_input.push_marker(timestamp, dir_true)
        time.sleep(window_size)
        app.gui.set_arrow(None)
        app.gui.set_text('')
        time.sleep(2)  # Downtime

    app.gui.kill()
    time.sleep(5)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = Path('C:/datasets/wearable-sensing') / timestamp
    path.mkdir(parents=True, exist_ok=False)

    n_repeats = 4
    window_size = 10
    App(
        fullscreen=True,
        experiment_func=partial(main, path=path, window_size=window_size, n_repeats=n_repeats),
    ).loop()
